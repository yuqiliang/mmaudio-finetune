"""Central paths for the non-stride Colab fine-tuning workflow.

Code runs from ``/content/MMAudio``. Datasets, manifests, latents, weights,
checkpoints, logs, and evaluation outputs stay in Google Drive. Set
``MMAUDIO_DRIVE_ROOT`` when a Drive uses a non-default layout.
"""

from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(os.environ.get("MMAUDIO_REPO_ROOT", "/content/MMAudio"))
MYDRIVE_ROOT = Path(os.environ.get("MMAUDIO_MYDRIVE_ROOT", "/content/drive/MyDrive"))


def _detect_drive_root() -> Path:
    configured = os.environ.get("MMAUDIO_DRIVE_ROOT")
    if configured:
        return Path(configured)

    candidates = [
        MYDRIVE_ROOT / "MMAudio_Yuqi",
        MYDRIVE_ROOT / "fine-tune",
    ]
    for candidate in candidates:
        if any(
            marker.exists()
            for marker in (
                candidate / "data",
                candidate / "weights",
                candidate / "latents",
                candidate / "outputs",
            )
        ):
            return candidate
    return candidates[0]


DRIVE_ROOT = _detect_drive_root()
DATA_DIR = DRIVE_ROOT / "data"
VIDEO_DIR = DATA_DIR / "soundscape_nonoverlap_8s"
MANIFEST_DIR = VIDEO_DIR / "metadata"
FINAL_MANIFEST = MANIFEST_DIR / "synced_clips_manifest.csv"

LATENT_DIR = DRIVE_ROOT / "latents" / "soundscape_nonoverlap_small_44k"
TRAIN_LATENT_DIR = LATENT_DIR / "train"
VAL_LATENT_DIR = LATENT_DIR / "val"
TEST_LATENT_DIR = LATENT_DIR / "test"

WEIGHTS_DIR = DRIVE_ROOT / "weights"
EXT_WEIGHTS_DIR = DRIVE_ROOT / "ext_weights"
OUTPUT_DIR = DRIVE_ROOT / "outputs"
EVAL_DIR = DRIVE_ROOT / "eval_outputs"
LOG_DIR = DRIVE_ROOT / "logs"
TSV_DIR = MANIFEST_DIR
CACHE_DIR = DRIVE_ROOT / "cache"
HF_CACHE_DIR = CACHE_DIR / "huggingface"
TORCH_CACHE_DIR = CACHE_DIR / "torch"

MODEL_WEIGHTS = WEIGHTS_DIR / "mmaudio_small_44k.pth"
VAE_WEIGHTS = EXT_WEIGHTS_DIR / "v1-44.pth"
SYNCHFORMER_WEIGHTS = EXT_WEIGHTS_DIR / "synchformer_state_dict.pth"

MODE = "44k"
MODEL = "small_44k"
AUDIO_SR = 44100
LATENT_SEQ_LEN = 345
CLIP_SEQ_LEN = 64
SYNC_SEQ_LEN = 192
TEXT_SEQ_LEN = 77

CLIP_SECONDS = 8
TEXT_LABEL = "urban soundscape"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")

def split_manifests() -> dict[str, Path]:
    return {
        split: MANIFEST_DIR / f"synced_{split}_clips_manifest.csv"
        for split in ("train", "val", "test")
    }


def split_latent_dirs() -> dict[str, Path]:
    return {
        "train": TRAIN_LATENT_DIR,
        "val": VAL_LATENT_DIR,
        "test": TEST_LATENT_DIR,
    }


def split_tsvs() -> dict[str, Path]:
    return {
        split: TSV_DIR / f"video_{split}_ft.tsv"
        for split in ("train", "val", "test")
    }


def ensure_drive_dirs() -> None:
    for path in [
        DRIVE_ROOT,
        DATA_DIR,
        VIDEO_DIR,
        MANIFEST_DIR,
        TRAIN_LATENT_DIR,
        VAL_LATENT_DIR,
        TEST_LATENT_DIR,
        WEIGHTS_DIR,
        EXT_WEIGHTS_DIR,
        OUTPUT_DIR,
        EVAL_DIR,
        LOG_DIR,
        HF_CACHE_DIR,
        TORCH_CACHE_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
