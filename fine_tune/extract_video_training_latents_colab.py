import os
import sys
import json
import argparse
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from tensordict import TensorDict

sys.path.append(".")

from fine_tune.custom_video_dataset import CustomVideoDataset
from mmaudio.model.utils.features_utils import FeaturesUtils


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Directory containing video clips"
    )
    parser.add_argument(
        "--latent_dir",
        type=str,
        required=True,
        help="Directory to save extracted latents"
    )

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--debug_limit",
        type=int,
        default=None,
        help="Only process the first N videos for debugging"
    )

    parser.add_argument(
        "--mode",
        type=str,
        default="16k",
        choices=["16k", "44k"],
        help="MMAudio mode"
    )
    parser.add_argument(
        "--vae_path",
        type=str,
        required=True,
        help="Path to VAE checkpoint"
    )
    parser.add_argument(
        "--bigvgan_path",
        type=str,
        default=None,
        help="Path to BigVGAN / vocoder checkpoint if needed"
    )
    parser.add_argument(
        "--synchformer_ckpt",
        type=str,
        default="./ext_weights/synchformer_state_dict.pth",
        help="Path to Synchformer checkpoint"
    )
    parser.add_argument(
        "--audio_sr",
        type=int,
        default=None,
        help="Target audio sampling rate; 16000 for 16k, 44100 for 44k"
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="training_latents",
        help="Name of saved memmap folder"
    )

    parser.add_argument("--clip_frame_size", type=int, default=384)
    parser.add_argument("--sync_frame_size", type=int, default=224)

    return parser.parse_args()


def build_feature_extractor(args, device):
    feature_extractor = FeaturesUtils(
        tod_vae_ckpt=args.vae_path,
        enable_conditions=True,
        bigvgan_vocoder_ckpt=args.bigvgan_path,
        synchformer_ckpt=args.synchformer_ckpt,
        mode=args.mode,
    ).eval().to(device)

    return feature_extractor


def save_metadata(save_path: Path, metadata: dict):
    meta_file = save_path / "metadata.json"
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def build_metadata(
    args,
    dataset_size,
    write_index,
    failed_batches,
    first_batch_shapes=None,
    expected_shapes=None,
):
    metadata = {
        "dataset_size": dataset_size,
        "written_samples": write_index,
        "mode": args.mode,
        "audio_sr": args.audio_sr,
        "video_dir": args.video_dir,
        "save_name": args.save_name,
        "failed_batches": failed_batches,
        "num_failed_batches": len(failed_batches),
    }

    if first_batch_shapes is not None:
        metadata["first_batch_shapes"] = first_batch_shapes

    if expected_shapes is not None:
        metadata["expected_shapes"] = expected_shapes

    return metadata


@torch.inference_mode()
def main():
    args = parse_args()

    os.makedirs(args.latent_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    if args.audio_sr is None:
        args.audio_sr = 16000 if args.mode == "16k" else 44100

    dataset = CustomVideoDataset(
        video_dir=args.video_dir,
        debug_limit=args.debug_limit,
        audio_sr=args.audio_sr,
        clip_frame_size=args.clip_frame_size,
        sync_frame_size=args.sync_frame_size,
    )

    dataset_size = len(dataset)
    print(f"Dataset size: {dataset_size}")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    feature_extractor = build_feature_extractor(args, device)

    save_path = Path(args.latent_dir) / args.save_name
    save_path.mkdir(parents=True, exist_ok=True)

    memmap_td = None
    write_index = 0
    failed_batches = []
    first_batch_shapes = None

    expected_shapes = {
        "clip_video_frames": 64,
        "sync_video_frames": 200,      # official 8s * 25fps input to sync encoder
        "clip_feature_seq_len": 64,
        "sync_feature_seq_len": 192,   # official sync output length
        "text_feature_seq_len": 77,
    }

    pbar = tqdm(loader, desc="Extracting")

    for batch_idx, batch in enumerate(pbar):
        try:
            clip_video = batch["clip_video"].to(device, non_blocking=True)
            sync_video = batch["sync_video"].to(device, non_blocking=True)
            audio = batch["audio"].to(device, non_blocking=True)
            text = batch["text"]
            paths = batch["path"]

            if audio.ndim == 3 and audio.shape[1] == 1:
                audio = audio.squeeze(1)  # [B, T]

            print(f"\nBatch {batch_idx}")
            print("clip_video:", clip_video.shape, clip_video.dtype, clip_video.device)
            print("sync_video:", sync_video.shape, sync_video.dtype, sync_video.device)
            print("audio:", audio.shape, audio.dtype, audio.device)
            print("text sample:", text[0] if isinstance(text, (list, tuple)) else text)
            print("example path:", paths[0])

            # Safety checks before encoding
            if clip_video.ndim != 5:
                raise ValueError(f"clip_video should be 5D [B, T, C, H, W], got {clip_video.shape}")

            if sync_video.ndim != 5:
                raise ValueError(f"sync_video should be 5D [B, T, C, H, W], got {sync_video.shape}")

            if clip_video.shape[1] != expected_shapes["clip_video_frames"]:
                raise AssertionError(
                    f"clip_video frames mismatch: got {clip_video.shape[1]}, "
                    f"expected {expected_shapes['clip_video_frames']}"
                )

            if sync_video.shape[1] != expected_shapes["sync_video_frames"]:
                raise AssertionError(
                    f"sync_video frames mismatch: got {sync_video.shape[1]}, "
                    f"expected {expected_shapes['sync_video_frames']}. "
                    f"Check custom_video_dataset.py sync_num_frames."
                )

            # 1) Audio
            print("Encoding audio...")
            audio_dist = feature_extractor.encode_audio(audio)
            print("audio ok")

            # 2) CLIP video
            print("Encoding CLIP video...")
            clip_features = feature_extractor.encode_video_with_clip(clip_video)
            print("clip ok")

            if clip_features.shape[1] != expected_shapes["clip_feature_seq_len"]:
                raise AssertionError(
                    f"clip_features length mismatch: got {clip_features.shape}, "
                    f"expected seq len {expected_shapes['clip_feature_seq_len']}"
                )

            # 3) Synchformer video
            print("Encoding Synchformer video...")
            sync_features = feature_extractor.encode_video_with_sync(sync_video)
            print("sync ok")

            if sync_features.shape[1] != expected_shapes["sync_feature_seq_len"]:
                raise AssertionError(
                    f"sync_features length mismatch: got {sync_features.shape}, "
                    f"expected seq len {expected_shapes['sync_feature_seq_len']}. "
                    f"This usually means sync_video input frames are wrong."
                )

            # 4) Text
            print("Encoding text...")
            if isinstance(text, tuple):
                text = list(text)
            elif isinstance(text, str):
                text = [text]

            text_features = feature_extractor.encode_text(list(text))
            print("text ok")

            if text_features.shape[1] != expected_shapes["text_feature_seq_len"]:
                print(
                    f"Warning: text_features seq len is {text_features.shape[1]}, "
                    f"expected {expected_shapes['text_feature_seq_len']}"
                )

            # Convert to CPU / contiguous for memmap write
            mean = audio_dist.mean.detach().cpu().transpose(1, 2).contiguous()
            std = audio_dist.std.detach().cpu().transpose(1, 2).contiguous()
            clip_features = clip_features.detach().cpu().contiguous()
            sync_features = sync_features.detach().cpu().contiguous()
            text_features = text_features.detach().cpu().contiguous()

            bsz = mean.shape[0]

            if batch_idx == 0:
                first_batch_shapes = {
                    "mean": list(mean.shape),
                    "std": list(std.shape),
                    "clip_features": list(clip_features.shape),
                    "sync_features": list(sync_features.shape),
                    "text_features": list(text_features.shape),
                }

                print("\nFirst batch output debug:")
                print("mean:", mean.shape)
                print("std:", std.shape)
                print("clip_features:", clip_features.shape)
                print("sync_features:", sync_features.shape)
                print("text_features:", text_features.shape)

            # Allocate memmap once after first successful batch
            if memmap_td is None:
                example_td = TensorDict(
                    {
                        "mean": torch.zeros(
                            (dataset_size, *mean.shape[1:]),
                            dtype=mean.dtype
                        ),
                        "std": torch.zeros(
                            (dataset_size, *std.shape[1:]),
                            dtype=std.dtype
                        ),
                        "clip_features": torch.zeros(
                            (dataset_size, *clip_features.shape[1:]),
                            dtype=clip_features.dtype
                        ),
                        "sync_features": torch.zeros(
                            (dataset_size, *sync_features.shape[1:]),
                            dtype=sync_features.dtype
                        ),
                        "text_features": torch.zeros(
                            (dataset_size, *text_features.shape[1:]),
                            dtype=text_features.dtype
                        ),
                    },
                    batch_size=[dataset_size],
                )

                memmap_td = example_td.memmap_(save_path)
                print(f"\nAllocated memmap at: {save_path}")

            end_index = write_index + bsz

            memmap_td["mean"][write_index:end_index] = mean
            memmap_td["std"][write_index:end_index] = std
            memmap_td["clip_features"][write_index:end_index] = clip_features
            memmap_td["sync_features"][write_index:end_index] = sync_features
            memmap_td["text_features"][write_index:end_index] = text_features

            write_index = end_index

            # Save metadata after each successful batch
            metadata = build_metadata(
                args=args,
                dataset_size=dataset_size,
                write_index=write_index,
                failed_batches=failed_batches,
                first_batch_shapes=first_batch_shapes,
                expected_shapes=expected_shapes,
            )
            save_metadata(save_path, metadata)

            if device == "cuda":
                torch.cuda.empty_cache()

            pbar.set_postfix({"written": write_index})

        except Exception as e:
            print(f"\nError in batch {batch_idx}")
            print("Exception repr:", repr(e))
            traceback.print_exc()

            failed_batches.append(
                {
                    "batch_idx": batch_idx,
                    "paths": list(batch["path"]) if "path" in batch else [],
                    "error": repr(e),
                }
            )

            metadata = build_metadata(
                args=args,
                dataset_size=dataset_size,
                write_index=write_index,
                failed_batches=failed_batches,
                first_batch_shapes=first_batch_shapes,
                expected_shapes=expected_shapes,
            )
            save_metadata(save_path, metadata)
            continue

    if memmap_td is None or write_index == 0:
        raise RuntimeError("No data was successfully processed.")

    print(f"\nStreaming memmap saved to: {save_path}")
    print(f"Successfully written samples: {write_index}")
    print(f"Failed batches: {len(failed_batches)}")

    summary = {
        "dataset_size": dataset_size,
        "written_samples": write_index,
        "failed_batches": failed_batches,
        "first_batch_shapes": first_batch_shapes,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()