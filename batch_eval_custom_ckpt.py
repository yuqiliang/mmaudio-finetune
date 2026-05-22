import argparse
import logging
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from mmaudio.eval_utils import (
    all_model_cfg,
    generate,
    load_video,
    make_video,
)
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils


log = logging.getLogger("batch_eval_custom_ckpt")
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)-8s]: %(message)s"
)


VIDEO_EXTENSIONS = [".mp4", ".mov", ".avi", ".wmv", ".mkv", ".webm"]


def load_custom_state(net, ckpt_path, device):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # 尽量兼容不同 checkpoint 格式
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            state_dict = ckpt["model"]
        elif "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        else:
            state_dict = ckpt
    else:
        state_dict = ckpt

    # 去掉可能的 module. 前缀
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[len("module."):]] = v
        else:
            cleaned_state_dict[k] = v

    missing, unexpected = net.load_state_dict(cleaned_state_dict, strict=False)

    log.info(f"Loaded checkpoint: {ckpt_path}")
    log.info(f"Missing keys: {len(missing)}")
    log.info(f"Unexpected keys: {len(unexpected)}")

    return missing, unexpected


def resolve_video_path(video_root: Path, rel_path: str) -> Path:
    """
    兼容 TSV 里只有 stem、没有扩展名的情况。
    例如:
      PancrasLock1_EQR_720p_0047
    自动匹配成:
      PancrasLock1_EQR_720p_0047.mp4
    """
    rel_path = rel_path.strip()
    base_path = video_root / rel_path

    # 1) 直接存在
    if base_path.exists():
        return base_path

    # 2) 如果 rel_path 自己已经带扩展名但文件不存在，直接返回原路径
    if Path(rel_path).suffix:
        return base_path

    # 3) 尝试常见视频扩展名
    for ext in VIDEO_EXTENSIONS:
        candidate = base_path.with_suffix(ext)
        if candidate.exists():
            return candidate

    # 4) 再做一次宽松匹配：同 stem 的任意文件
    matches = []
    for p in video_root.glob(f"{rel_path}.*"):
        if p.is_file():
            matches.append(p)

    if len(matches) == 1:
        return matches[0]

    # 找不到就返回原路径，后面主循环会 warning + skip
    return base_path


def read_test_tsv(tsv_path, video_root):
    tsv_path = Path(tsv_path)
    video_root = Path(video_root)

    if not tsv_path.exists():
        raise FileNotFoundError(f"TSV not found: {tsv_path}")
    if not video_root.exists():
        raise FileNotFoundError(f"Video root not found: {video_root}")

    samples = []

    with open(tsv_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if len(lines) == 0:
        raise RuntimeError(f"Empty TSV: {tsv_path}")

    # 兼容 header / no-header
    header = lines[0].split("\t")
    has_header = any(
        x.lower() in {"id", "path", "video", "label", "text", "caption"}
        for x in header
    )

    start_idx = 1 if has_header else 0

    for line in lines[start_idx:]:
        parts = line.split("\t")

        if len(parts) == 1:
            rel_path = parts[0]
            caption = "urban soundscape"
        else:
            rel_path = parts[0]
            caption = parts[1].strip() if parts[1].strip() else "urban soundscape"

        video_path = resolve_video_path(video_root, rel_path)
        samples.append((video_path, caption))

    return samples


def save_audio_wav(save_path: Path, audio: torch.Tensor, sampling_rate: int):
    """
    用 soundfile 保存，绕开 torchaudio.save -> torchcodec 的兼容问题。
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    audio_np = audio.detach().cpu().float()

    # 常见情况：audio shape = [1, T] 或 [T]
    if audio_np.ndim == 2:
        if audio_np.shape[0] == 1:
            audio_np = audio_np.squeeze(0)
        else:
            # 多声道时转成 [T, C]
            audio_np = audio_np.transpose(0, 1)

    audio_np = audio_np.numpy()
    audio_np = np.asarray(audio_np, dtype=np.float32)

    sf.write(str(save_path), audio_np, sampling_rate)
    log.info(f"Saved: {save_path}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--test_tsv", type=Path, required=True)
    parser.add_argument("--video_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg_strength", type=float, default=4.5)
    parser.add_argument("--num_steps", type=int, default=25)
    parser.add_argument("--full_precision", action="store_true")
    parser.add_argument("--skip_video_composite", action="store_true")

    args = parser.parse_args()

    if args.variant not in all_model_cfg:
        raise ValueError(
            f"Unknown variant: {args.variant}. "
            f"Available: {list(all_model_cfg.keys())}"
        )

    model = all_model_cfg[args.variant]
    model.download_if_needed()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.full_precision else torch.bfloat16

    log.info(f"Using device: {device}")
    log.info(f"Using dtype: {dtype}")

    net = get_my_mmaudio(model.model_name).to(device, dtype).eval()
    missing, unexpected = load_custom_state(net, args.weights, device)

    if len(missing) > 20 or len(unexpected) > 20:
        raise RuntimeError(
            f"Checkpoint structure mismatch: missing={len(missing)}, unexpected={len(unexpected)}"
        )

    fm = FlowMatching(
        min_sigma=0,
        inference_mode="euler",
        num_steps=args.num_steps,
    )

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=model.vae_path,
        synchformer_ckpt=model.synchformer_ckpt,
        enable_conditions=True,
        mode=model.mode,
        bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
        need_vae_encoder=False,
    ).to(device, dtype).eval()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    samples = read_test_tsv(args.test_tsv, args.video_root)
    log.info(f"Found {len(samples)} test samples")

    processed = 0
    skipped = 0

    for idx, (video_path, caption) in enumerate(samples, start=1):
        log.info(f"[{idx}/{len(samples)}] Processing: {video_path.name}")

        if not video_path.exists():
            log.warning(f"Skip missing video: {video_path}")
            skipped += 1
            continue

        try:
            rng = torch.Generator(device=device)
            rng.manual_seed(args.seed)

            video_info = load_video(video_path, args.duration)

            # ── 强制使用 CLI 指定的 duration（8.0），不用 video_info 的 7.96 ──
            forced_duration = args.duration  # 默认 8.0

            clip_frames = video_info.clip_frames
            sync_frames = video_info.sync_frames

            # ── Pad sync_frames → 目标帧数 (25fps × duration) ──
            target_sync = int(25 * forced_duration)   # 200 for 8s
            log.info(f"sync_frames raw shape = {tuple(sync_frames.shape)}, target = {target_sync}")
            if sync_frames.shape[0] < target_sync:
                pad_n = target_sync - sync_frames.shape[0]
                sync_frames = torch.cat(
                    [sync_frames, sync_frames[-1:].repeat(pad_n, 1, 1, 1)], dim=0
                )
            elif sync_frames.shape[0] > target_sync:
                sync_frames = sync_frames[:target_sync]

            # ── Pad clip_frames → 目标帧数 (8fps × duration) ──
            target_clip = int(8 * forced_duration)    # 64 for 8s
            if clip_frames is not None:
                log.info(f"clip_frames raw shape = {tuple(clip_frames.shape)}, target = {target_clip}")
                if clip_frames.shape[0] < target_clip:
                    pad_n = target_clip - clip_frames.shape[0]
                    clip_frames = torch.cat(
                        [clip_frames, clip_frames[-1:].repeat(pad_n, 1, 1, 1)], dim=0
                    )
                elif clip_frames.shape[0] > target_clip:
                    clip_frames = clip_frames[:target_clip]

            # ── 验证 pad 后的实际形状 ──
            log.info(f"clip_frames after pad = {tuple(clip_frames.shape) if clip_frames is not None else None}")
            log.info(f"sync_frames after pad = {tuple(sync_frames.shape)}")

            if clip_frames is not None:
                clip_frames = clip_frames.unsqueeze(0)
            sync_frames = sync_frames.unsqueeze(0)

            # ── 用强制 duration 设置 seq lengths ──
            seq_cfg = model.seq_cfg
            seq_cfg.duration = forced_duration        # 8.0，不是 7.96

            net.update_seq_lengths(
                seq_cfg.latent_seq_len,
                seq_cfg.clip_seq_len,
                seq_cfg.sync_seq_len,
            )
            log.info(f"seq_cfg: clip_seq_len={seq_cfg.clip_seq_len}, "
                      f"sync_seq_len={seq_cfg.sync_seq_len}, "
                      f"latent_seq_len={seq_cfg.latent_seq_len}")

            audios = generate(
                clip_frames,
                sync_frames,
                [caption],
                negative_text=[""],
                feature_utils=feature_utils,
                net=net,
                fm=fm,
                rng=rng,
                cfg_strength=args.cfg_strength,
            )

            audio = audios.float().cpu()[0]

            # 直接保存 wav，后面 evaluate 更方便
            save_path = args.output_dir / f"{video_path.stem}.wav"
            save_audio_wav(save_path, audio, seq_cfg.sampling_rate)

            if not args.skip_video_composite:
                video_save_path = args.output_dir / f"{video_path.stem}.mp4"
                make_video(
                    video_info,
                    video_save_path,
                    audio,
                    sampling_rate=seq_cfg.sampling_rate,
                )
                log.info(f"Video saved to {video_save_path}")

            if torch.cuda.is_available():
                log.info(
                    "Memory usage: %.2f GB",
                    torch.cuda.max_memory_allocated() / (2 ** 30)
                )

            processed += 1

        except Exception as e:
            log.exception(f"Failed on {video_path}: {e}")
            skipped += 1
            continue

    log.info(f"Done. processed={processed}, skipped={skipped}, total={len(samples)}")


if __name__ == "__main__":
    main()