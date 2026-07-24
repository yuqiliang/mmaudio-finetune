import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.io import read_video


class CustomVideoDataset(Dataset):
    """
    Dataset for urban video clips used in MMAudio feature extraction.

    Official MMAudio 8s setting:
        - clip branch:  8 fps  × 8s = 64 frames
        - sync branch: 25 fps × 8s = 200 frames

    Returns a dict with:
        - clip_video: [clip_num_frames, 3, H, W], float32 in [0, 1]
        - sync_video: [sync_num_frames, 3, H, W], float32 in [0, 1]
        - audio: [1, T]
        - text: str
        - path: str
        - video_fps: float
        - audio_fps: int
    """

    def __init__(
        self,
        video_dir: Optional[str] = None,
        video_paths: Optional[Sequence[Union[str, Path]]] = None,
        text_label: str = "urban soundscape",
        duration_seconds: float = 8.0,
        clip_num_frames: int = 64,
        sync_num_frames: int = 200,   # official: 8s * 25fps
        clip_frame_size: int = 384,
        sync_frame_size: int = 224,
        audio_sr: Optional[int] = None,
        extensions: Optional[List[str]] = None,
        debug_limit: Optional[int] = None,
    ):
        super().__init__()

        if (video_dir is None) == (video_paths is None):
            raise ValueError("Provide exactly one of video_dir or video_paths")

        self.video_dir = video_dir
        self.text_label = text_label
        self.duration_seconds = duration_seconds
        self.clip_num_frames = clip_num_frames
        self.sync_num_frames = sync_num_frames
        self.clip_frame_size = clip_frame_size
        self.sync_frame_size = sync_frame_size
        self.audio_sr = audio_sr

        if extensions is None:
            extensions = ["mp4", "mov", "mkv", "avi", "webm", "m4v"]

        if video_paths is not None:
            self.video_paths = [str(Path(path).expanduser().resolve()) for path in video_paths]
        else:
            assert video_dir is not None
            discovered = []
            for ext in extensions:
                discovered.extend(glob.glob(os.path.join(video_dir, f"*.{ext}")))
                discovered.extend(glob.glob(os.path.join(video_dir, f"*.{ext.upper()}")))
            self.video_paths = sorted(set(discovered))

        if debug_limit is not None:
            self.video_paths = self.video_paths[:debug_limit]

        if len(self.video_paths) == 0:
            raise FileNotFoundError(f"No video files found in: {video_dir or 'manifest'}")
        if len(self.video_paths) != len(set(self.video_paths)):
            raise ValueError("Video paths must be unique")

        missing = [path for path in self.video_paths if not Path(path).is_file()]
        if missing:
            raise FileNotFoundError(f"Manifest-listed video does not exist: {missing[0]}")
        print(f"Found {len(self.video_paths)} video files")

    def __len__(self) -> int:
        return len(self.video_paths)

    def _safe_read_video(self, path: str) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        try:
            video, audio, info = read_video(path, pts_unit="sec")
            return video, audio, info
        except Exception as e:
            raise RuntimeError(f"Failed to read video {path}: {e}")

    def _sample_frames(
        self,
        video: torch.Tensor,
        num_frames: int,
        frame_size: int,
    ) -> torch.Tensor:
        total_frames = video.shape[0]

        if total_frames == 0:
            raise ValueError("Video has 0 frames.")

        if total_frames == 1:
            indices = torch.zeros(num_frames, dtype=torch.long)
        else:
            indices = torch.linspace(0, total_frames - 1, steps=num_frames)
            indices = indices.round().long().clamp(0, total_frames - 1)

        frames = video[indices]  # [N, H, W, C]
        frames = frames.permute(0, 3, 1, 2).float() / 255.0  # [N, C, H, W]

        frames = F.interpolate(
            frames,
            size=(frame_size, frame_size),
            mode="bilinear",
            align_corners=False,
        )

        return frames.contiguous()

    def _resample_audio(
        self,
        audio: torch.Tensor,
        orig_sr: int,
        target_sr: int,
    ) -> torch.Tensor:
        if orig_sr == target_sr:
            return audio

        old_len = audio.shape[-1]
        new_len = max(1, int(round(old_len * target_sr / orig_sr)))

        audio = audio.unsqueeze(0)  # [1, 1, T]
        audio = F.interpolate(
            audio,
            size=new_len,
            mode="linear",
            align_corners=False,
        )
        audio = audio.squeeze(0)  # [1, T]
        return audio

    def _process_audio(
        self,
        audio: torch.Tensor,
        info: Dict[str, Any],
    ) -> Tuple[torch.Tensor, int]:
        audio_fps = info.get("audio_fps", None)

        if audio.numel() == 0:
            fallback_sr = self.audio_sr if self.audio_sr is not None else 16000
            target_samples = round(fallback_sr * self.duration_seconds)
            silent = torch.zeros(1, target_samples, dtype=torch.float32)
            return silent, fallback_sr

        audio = audio.float()

        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim == 2:
            if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
                audio = audio.mean(dim=0, keepdim=True)
            elif audio.shape[1] <= 8 and audio.shape[0] > audio.shape[1]:
                audio = audio.mean(dim=1).unsqueeze(0)
            else:
                audio = audio.reshape(1, -1)
        else:
            audio = audio.reshape(1, -1)

        if audio_fps is None:
            audio_fps = self.audio_sr if self.audio_sr is not None else 16000

        if self.audio_sr is not None and audio_fps != self.audio_sr:
            audio = self._resample_audio(audio, audio_fps, self.audio_sr)
            audio_fps = self.audio_sr

        target_samples = round(audio_fps * self.duration_seconds)
        if audio.shape[-1] < target_samples:
            audio = F.pad(audio, (0, target_samples - audio.shape[-1]))
        else:
            audio = audio[..., :target_samples]

        return audio.contiguous(), int(audio_fps)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.video_paths[idx]

        raw_video, raw_audio, info = self._safe_read_video(path)
        video_fps = info.get("video_fps", 0.0)

        clip_video = self._sample_frames(
            raw_video,
            num_frames=self.clip_num_frames,
            frame_size=self.clip_frame_size,
        )

        sync_video = self._sample_frames(
            raw_video,
            num_frames=self.sync_num_frames,
            frame_size=self.sync_frame_size,
        )

        # official expected input lengths
        assert clip_video.shape[0] == 64, f"clip_video frames={clip_video.shape[0]}, expected 64"
        assert sync_video.shape[0] == 200, f"sync_video frames={sync_video.shape[0]}, expected 200"

        audio, audio_fps = self._process_audio(raw_audio, info)

        if idx == 0:
            print("DEBUG in dataset:")
            print("path:", path)
            print("raw_video shape:", tuple(raw_video.shape))
            print("raw_audio shape:", tuple(raw_audio.shape))
            print("processed audio shape:", tuple(audio.shape))
            print("clip_video shape:", tuple(clip_video.shape))
            print("sync_video shape:", tuple(sync_video.shape))
            print("video_fps:", video_fps)
            print("audio_fps:", audio_fps)

        item = {
            "clip_video": clip_video,
            "sync_video": sync_video,
            "audio": audio,
            "text": self.text_label,
            "path": path,
            "video_fps": float(video_fps),
            "audio_fps": int(audio_fps),
        }

        return item
