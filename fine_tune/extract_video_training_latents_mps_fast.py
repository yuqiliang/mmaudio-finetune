#!/usr/bin/env python3
"""
extract_video_training_latents_mps_fast.py
Faster Mac-friendly feature extraction for MMAudio fine-tuning.
caffeinate python3 fine_tune/extract_video_training_latents_mps_fast.py \
  --latent_dir ./output/latents_debug \
  --output_dir ./output/memmap_debug \
  --debug_limit 16 \
  --batch_size 2
"""

import gc
import logging
import os
import sys
from argparse import ArgumentParser
from datetime import timedelta
from pathlib import Path

import pandas as pd
import tensordict as td
import torch
import torch.distributed as distributed
from torch.utils.data import DataLoader, Subset
from torch.utils.data._utils.collate import default_collate
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from fine_tune.mac_video_dataset import MacVideoDataset
from mmaudio.model.utils.features_utils import FeaturesUtils

# =========================================================
# MODEL CONFIGURATION
# =========================================================

SAMPLING_RATE = 16000
DURATION_SEC = 8.0
NUM_SAMPLES = 128000
vae_path = './ext_weights/v1-16.pth'
bigvgan_path = './ext_weights/best_netG.pt'
mode = '16k'

# Uncomment below for 44.1kHz model
# SAMPLING_RATE = 44100
# DURATION_SEC = 8.0
# NUM_SAMPLES = 353280
# vae_path = './ext_weights/v1-44.pth'
# bigvgan_path = None
# mode = '44k'

synchformer_ckpt = './ext_weights/synchformer_state_dict.pth'

# =========================================================
# DEFAULT MAC SETTINGS
# =========================================================

DEFAULT_BATCH_SIZE = 8
DEFAULT_NUM_WORKERS = 0
DEFAULT_MEMORY_CHECK_INTERVAL = 100

# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
log = logging.getLogger(__name__)

# =========================================================
# DATA CONFIG
# =========================================================

data_cfg = {
    'my_videos': {
        'root': '/Users/yuqiliang/Documents/UCLPhD/pilot_study/output_split_stride10',
        'subset_name': './fine_tune/training/video_train_stride_10.tsv',
        'normalize_audio': True,
    },
}

# =========================================================
# HELPERS
# =========================================================

def safe_collate(batch):
    batch = [x for x in batch if x is not None]
    if len(batch) == 0:
        return None
    return default_collate(batch)


def check_system_resources():
    log.info("=== System Resources Check ===")
    try:
        import multiprocessing
        log.info(f"CPU cores: {multiprocessing.cpu_count()}")
    except Exception:
        log.info("CPU cores: Unable to determine")

    if torch.backends.mps.is_available():
        log.info("GPU: MPS (Metal Performance Shaders) available")
    elif torch.cuda.is_available():
        log.info("GPU: CUDA available")
    else:
        log.info("GPU: None available, using CPU")
    log.info("================================")


def get_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("✅ Using MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("✅ Using CUDA")
    else:
        device = torch.device("cpu")
        log.info("⚠️ Using CPU (will be slow)")
    return device


def enable_runtime_optimizations():
    try:
        torch.set_float32_matmul_precision("high")
        log.info("✅ torch float32 matmul precision = high")
    except Exception as e:
        log.warning(f"Could not set matmul precision: {e}")


def clear_memory_cache():
    if torch.backends.mps.is_available():
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    elif torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    gc.collect()


def distributed_setup():
    try:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            distributed.init_process_group(
                backend=backend,
                timeout=timedelta(hours=1)
            )
            rank = distributed.get_rank()
            world_size_val = distributed.get_world_size()
            log.info(f'Distributed mode: rank={rank}, world_size={world_size_val}')
            return rank, world_size_val
        else:
            log.info('Single device mode (no distributed training)')
            return 0, 1
    except Exception as e:
        log.warning(f"Distributed setup failed: {e}. Falling back to single device mode.")
        return 0, 1


def create_tsv_file(video_files, tsv_path: Path):
    log.info(f"Creating TSV file automatically: {tsv_path}")
    rows = []
    for video_path in sorted(video_files):
        rows.append({
            "id": video_path.stem,
            "label": "audio from video"
        })
    df = pd.DataFrame(rows)
    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(tsv_path, sep='\t', index=False)
    log.info(f"✅ Created TSV file with {len(df)} entries")


def verify_data_config():
    log.info("=== Verifying Data Configuration ===")
    for split_name, config in data_cfg.items():
        log.info(f"Checking dataset: {split_name}")

        video_dir = Path(config['root'])
        if not video_dir.exists():
            log.error(f"❌ Video directory not found: {video_dir}")
            return False

        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm']
        video_files = []
        for ext in video_extensions:
            video_files.extend(list(video_dir.glob(f'*{ext}')))
            video_files.extend(list(video_dir.glob(f'*{ext.upper()}')))

        if not video_files:
            log.error(f"❌ No video files found in {video_dir}")
            return False

        log.info(f"  Found {len(video_files)} video files")

        tsv_path = Path(config['subset_name'])
        if not tsv_path.exists():
            log.warning(f"TSV file not found: {tsv_path}")
            create_tsv_file(video_files, tsv_path)
        else:
            df = pd.read_csv(tsv_path, sep='\t')
            log.info(f"  TSV file: {len(df)} entries")
            if 'id' not in df.columns or 'label' not in df.columns:
                log.error("❌ TSV file must contain 'id' and 'label'")
                return False

    log.info("✅ Data configuration verified")
    return True


def setup_dataset(split: str, rank: int, world_size_val: int, debug_limit: int, batch_size: int, num_workers: int):
    log.info(f"Setting up dataset for split: {split}")

    dataset = MacVideoDataset(
        root=data_cfg[split]['root'],
        tsv_path=data_cfg[split]['subset_name'],
        sample_rate=SAMPLING_RATE,
        duration_sec=DURATION_SEC,
        audio_samples=NUM_SAMPLES,
        normalize_audio=data_cfg[split]['normalize_audio'],
    )

    original_len = len(dataset)
    if debug_limit and debug_limit > 0:
        debug_limit = min(debug_limit, original_len)
        dataset = Subset(dataset, list(range(debug_limit)))
        log.info(f"⚠️ Debug mode enabled: using first {debug_limit}/{original_len} samples")

    if world_size_val > 1:
        sampler = DistributedSampler(dataset, rank=rank, shuffle=False)
        log.info(f"Using DistributedSampler: rank={rank}, world_size={world_size_val}")
    else:
        sampler = None
        log.info("Using regular sampling")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        collate_fn=safe_collate,
        pin_memory=False,          # MPS下通常没什么明显帮助
        persistent_workers=False,  # num_workers=0时无效，保守起见
    )

    log.info(f"Dataset setup complete: {len(dataset)} samples, {len(loader)} batches")
    return dataset, loader


def encode_text_with_cache(feature_extractor, captions, text_cache):
    """
    captions: list[str]
    Return: tensor [B, ...]
    """
    uncached = []
    for c in captions:
        if c not in text_cache:
            uncached.append(c)

    if uncached:
        unique_uncached = list(dict.fromkeys(uncached))
        feats = feature_extractor.encode_text(unique_uncached).detach().cpu()
        for c, f in zip(unique_uncached, feats):
            text_cache[c] = f

    stacked = torch.stack([text_cache[c] for c in captions], dim=0)
    return stacked


@torch.inference_mode()
def extract():
    log.info("🚀 Starting MMAudio Feature Extraction")
    log.info("=" * 50)

    check_system_resources()
    enable_runtime_optimizations()

    if not verify_data_config():
        log.error("❌ Data configuration verification failed")
        sys.exit(1)

    rank, world_size_val = distributed_setup()

    parser = ArgumentParser(description="Extract video training features for MMAudio")
    parser.add_argument('--latent_dir', type=Path, default=Path('./output/latents'))
    parser.add_argument('--output_dir', type=Path, default=Path('./output/memmap'))
    parser.add_argument('--debug_limit', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument('--memory_check_interval', type=int, default=DEFAULT_MEMORY_CHECK_INTERVAL)
    args = parser.parse_args()

    latent_dir = args.latent_dir
    output_dir = args.output_dir
    debug_limit = args.debug_limit
    batch_size = args.batch_size
    num_workers = args.num_workers
    memory_check_interval = args.memory_check_interval

    log.info(f"📁 Latent directory: {latent_dir}")
    log.info(f"📁 Output directory: {output_dir}")
    log.info(f"⚙️ batch_size={batch_size}, num_workers={num_workers}, memory_check_interval={memory_check_interval}")

    device = get_device()

    log.info("🔧 Loading feature extractor models...")
    try:
        feature_extractor = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            enable_conditions=True,
            bigvgan_vocoder_ckpt=bigvgan_path,
            synchformer_ckpt=synchformer_ckpt,
            mode=mode
        ).eval().to(device)
        log.info("✅ Feature extractor loaded successfully")
    except Exception as e:
        log.error(f"❌ Failed to load feature extractor: {e}")
        sys.exit(1)

    any_success = False

    for split in data_cfg.keys():
        log.info(f"\n🎬 Processing split: {split}")
        log.info("=" * 30)

        this_latent_dir = latent_dir / split
        this_latent_dir.mkdir(parents=True, exist_ok=True)

        dataset, loader = setup_dataset(
            split=split,
            rank=rank,
            world_size_val=world_size_val,
            debug_limit=debug_limit,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        log.info(f"🔄 Starting feature extraction for {len(dataset)} samples...")

        batch_count = 0
        success_count = 0
        failed_batch_count = 0
        text_cache = {}

        for curr_iter, data in enumerate(tqdm(loader, desc=f"Extracting {split}")):
            try:
                if data is None:
                    log.warning(f"Skipping empty batch at iter {curr_iter}")
                    failed_batch_count += 1
                    continue

                output = {
                    'id': data['id'],
                    'caption': data['caption'],
                }

                # -------- audio --------
                audio = data['audio'].to(device, non_blocking=False)
                dist = feature_extractor.encode_audio(audio)
                output['mean'] = dist.mean.detach().cpu().transpose(1, 2).contiguous()
                output['std'] = dist.std.detach().cpu().transpose(1, 2).contiguous()
                del audio, dist

                # -------- clip video --------
                clip_video = data['clip_video'].to(device, non_blocking=False)
                clip_features = feature_extractor.encode_video_with_clip(clip_video)
                output['clip_features'] = clip_features.detach().cpu().contiguous()
                del clip_video, clip_features

                # -------- sync video --------
                sync_video = data['sync_video'].to(device, non_blocking=False)
                sync_features = feature_extractor.encode_video_with_sync(sync_video)
                output['sync_features'] = sync_features.detach().cpu().contiguous()
                del sync_video, sync_features

                # -------- text (cached) --------
                captions = list(data['caption'])
                text_features = encode_text_with_cache(feature_extractor, captions, text_cache)
                output['text_features'] = text_features.contiguous()

                output_file = this_latent_dir / f'r{rank}_{curr_iter}.pth'
                torch.save(output, output_file)

                batch_count += 1
                success_count += len(output['id'])
                any_success = True

                if batch_count % memory_check_interval == 0:
                    clear_memory_cache()
                    log.info(f"Processed {batch_count}/{len(loader)} batches | text_cache={len(text_cache)}")

            except Exception as e:
                failed_batch_count += 1
                log.exception(f"❌ Error processing batch {curr_iter}: {e}")
                clear_memory_cache()
                continue

        log.info(
            f"Extraction loop finished for {split}: "
            f"{success_count} samples saved, "
            f"{failed_batch_count} failed/empty batches"
        )

        if world_size_val > 1:
            distributed.barrier()

        if rank == 0:
            combine_results(this_latent_dir, output_dir, split)

    if any_success:
        log.info("\n🎉 Feature extraction completed with successful outputs.")
    else:
        log.error("\n❌ No features were successfully extracted.")


def combine_results(latent_dir: Path, output_dir: Path, split: str):
    log.info(f"📦 Combining results for {split}")

    used_id = set()
    list_of_ids_and_labels = []
    output_data = {
        'mean': [],
        'std': [],
        'clip_features': [],
        'sync_features': [],
        'text_features': [],
    }

    batch_files = sorted([f for f in os.listdir(latent_dir) if f.endswith('.pth')])
    log.info(f"Found {len(batch_files)} intermediate batch files")

    if len(batch_files) == 0:
        log.error("❌ No .pth batch files found. Skipping combine step.")
        return

    for filename in tqdm(batch_files, desc="Combining batches"):
        data = torch.load(latent_dir / filename, map_location='cpu', weights_only=True)
        batch_size = len(data['id'])

        for bi in range(batch_size):
            this_id = data['id'][bi]
            this_caption = data['caption'][bi]

            if this_id in used_id:
                continue

            list_of_ids_and_labels.append({
                'id': this_id,
                'label': this_caption,
            })
            used_id.add(this_id)

            output_data['mean'].append(data['mean'][bi])
            output_data['std'].append(data['std'][bi])
            output_data['clip_features'].append(data['clip_features'][bi])
            output_data['sync_features'].append(data['sync_features'][bi])
            output_data['text_features'].append(data['text_features'][bi])

    if not list_of_ids_and_labels:
        log.error("❌ No valid data found to combine.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = output_dir / f'finetune-{split}.tsv'
    output_df = pd.DataFrame(list_of_ids_and_labels)
    output_df.to_csv(tsv_path, sep='\t', index=False)
    log.info(f"📄 Saved metadata TSV: {tsv_path} ({len(output_df)} samples)")

    output_data = {k: torch.stack(v) for k, v in output_data.items()}
    tensor_dict = td.TensorDict(output_data)
    memmap_path = output_dir / f'finetune-{split}'
    tensor_dict.memmap_(memmap_path)

    log.info(f"✅ Memory-mapped tensors saved to: {memmap_path}")
    for key, tensor in output_data.items():
        log.info(f"  {key}: {tuple(tensor.shape)}")


if __name__ == '__main__':
    try:
        extract()
    except KeyboardInterrupt:
        log.info("🛑 Extraction interrupted by user")
        sys.exit(1)
    except Exception as e:
        log.exception(f"❌ Fatal error: {e}")
        sys.exit(1)
    finally:
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            try:
                distributed.destroy_process_group()
            except Exception:
                pass
        clear_memory_cache()
        log.info("✅ Feature extraction finished")