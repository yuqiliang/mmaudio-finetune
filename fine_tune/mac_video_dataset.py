#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
import subprocess
from pathlib import Path

import av
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class MacVideoDataset(Dataset):
    """
    Mac-friendly video dataset for MMAudio feature extraction.

    Returns:
        {
            'id': str,
            'caption': str,
            'audio': Tensor [audio_samples],
            'clip_video': Tensor [64, 3, 384, 384],   # 8 fps * 8 sec
            'sync_video': Tensor [192, 3, 224, 224],  # 24 fps * 8 sec
        }
    """

    def __init__(
        self,
        root,
        tsv_path,
        sample_rate=16000,
        duration_sec=8.0,
        audio_samples=128000,
        normalize_audio=True,
    ):
        self.root = Path(root)
        self.tsv_path = Path(tsv_path)
        self.sample_rate = sample_rate
        self.duration_sec = duration_sec
        self.audio_samples = audio_samples
        self.normalize_audio = normalize_audio

        self.clip_num_frames = 64
        self.clip_target_size = 384

        self.sync_num_frames = 192
        self.sync_target_size = 224

        self.df = pd.read_csv(self.tsv_path, sep="\t")
        if "id" not in self.df.columns or "label" not in self.df.columns:
            raise ValueError("TSV must contain 'id' and 'label' columns")

        valid_rows = []
        missing = []

        for _, row in self.df.iterrows():
            vid = str(row["id"])
            video_path = self._find_video_file(vid)
            if video_path is None:
                missing.append(vid)
            else:
                valid_rows.append(
                    {
                        "id": vid,
                        "label": str(row["label"]),
                        "path": video_path,
                    }
                )

        self.samples = valid_rows

        print(f"{len(self.samples)} valid videos found in {self.root}")
        print(f"{len(self.df)} entries found in {self.tsv_path}")
        print(f"{len(missing)} videos missing in {self.root}")

    def _find_video_file(self, vid):
        exts = [".mp4", ".mov", ".avi", ".mkv", ".webm", ".MP4", ".MOV", ".AVI", ".MKV", ".WEBM"]
        for ext in exts:
            p = self.root / f"{vid}{ext}"
            if p.exists():
                return p
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples[idx]
        vid = row["id"]
        caption = row["label"]
        video_path = row["path"]

        try:
            audio = self._load_audio_ffmpeg(video_path)

            meta = self._get_video_meta(video_path)

            clip_indices = self._make_frame_indices(
                total_frames=meta["total_frames"],
                num_frames=self.clip_num_frames,
            )
            sync_indices = self._make_frame_indices(
                total_frames=meta["total_frames"],
                num_frames=self.sync_num_frames,
            )

            needed_indices = sorted(set(clip_indices.tolist() + sync_indices.tolist()))
            decoded = self._decode_selected_frames(video_path, needed_indices)

            clip_frames = [decoded[i] for i in clip_indices]
            sync_frames = [decoded[i] for i in sync_indices]

            clip_video = self._process_clip_frames(clip_frames)
            sync_video = self._process_sync_frames(sync_frames)

            return {
                "id": vid,
                "caption": caption,
                "audio": audio,
                "clip_video": clip_video,
                "sync_video": sync_video,
            }

        except Exception as e:
            print(f"Error loading video {vid}: {e}")
            return None

    def _load_audio_ffmpeg(self, video_path):
        cmd = [
            "ffmpeg",
            "-v", "error",
            "-i", str(video_path),
            "-f", "f32le",
            "-acodec", "pcm_f32le",
            "-ac", "1",
            "-ar", str(self.sample_rate),
            "-"
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )

        audio = np.frombuffer(result.stdout, dtype=np.float32)
        audio = torch.from_numpy(audio.copy())

        if audio.numel() == 0:
            raise RuntimeError("decoded audio is empty")

        if self.normalize_audio:
            max_val = audio.abs().max()
            if max_val > 0:
                audio = audio / max_val

        if audio.numel() < self.audio_samples:
            pad = self.audio_samples - audio.numel()
            audio = F.pad(audio, (0, pad))
        else:
            audio = audio[: self.audio_samples]

        return audio

    def _get_video_meta(self, video_path):
        container = av.open(str(video_path))
        stream = container.streams.video[0]

        total_frames = stream.frames
        if total_frames is None or total_frames <= 0:
            # fallback: manually count
            total_frames = 0
            for _ in container.decode(video=0):
                total_frames += 1
            container.close()
            container = av.open(str(video_path))
        else:
            container.close()

        if total_frames <= 0:
            raise RuntimeError("no video frames found")

        return {"total_frames": total_frames}

    def _make_frame_indices(self, total_frames, num_frames):
        """
        Uniformly sample frame indices across available frames.
        Duplicate endpoints if video has very few frames.
        """
        if total_frames <= 0:
            raise RuntimeError("invalid total_frames")

        if total_frames == 1:
            return np.zeros(num_frames, dtype=np.int64)

        return np.linspace(
            0,
            total_frames - 1,
            num=num_frames,
            dtype=np.int64
        )

    def _decode_selected_frames(self, video_path, needed_indices):
        """
        Decode only frames whose indices are needed.
        Return dict: frame_idx -> RGB ndarray
        """
        needed_set = set(int(i) for i in needed_indices)
        decoded = {}

        container = av.open(str(video_path))
        stream = container.streams.video[0]

        for frame_idx, frame in enumerate(container.decode(stream)):
            if frame_idx in needed_set:
                decoded[frame_idx] = frame.to_rgb().to_ndarray()
                if len(decoded) == len(needed_set):
                    break

        container.close()

        if not decoded:
            raise RuntimeError("failed to decode selected frames")

        # backfill any missing index with nearest previous available frame
        available = sorted(decoded.keys())
        for idx in needed_indices:
            if idx not in decoded:
                nearest = min(available, key=lambda x: abs(x - idx))
                decoded[idx] = decoded[nearest]

        return decoded

    def _resize_square(self, img_np, size):
        pil = Image.fromarray(img_np)
        pil = pil.resize((size, size), Image.BICUBIC)
        arr = np.asarray(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _resize_shorter_edge_and_center_crop(self, img_np, target_size):
        pil = Image.fromarray(img_np)
        w, h = pil.size

        if w < h:
            new_w = target_size
            new_h = int(round(h * target_size / w))
        else:
            new_h = target_size
            new_w = int(round(w * target_size / h))

        pil = pil.resize((new_w, new_h), Image.BICUBIC)

        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        pil = pil.crop((left, top, left + target_size, top + target_size))

        arr = np.asarray(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _process_clip_frames(self, frames):
        processed = [self._resize_square(f, self.clip_target_size) for f in frames]
        return torch.stack(processed, dim=0)

    def _process_sync_frames(self, frames):
        processed = [
            self._resize_shorter_edge_and_center_crop(f, self.sync_target_size)
            for f in frames
        ]
        return torch.stack(processed, dim=0)