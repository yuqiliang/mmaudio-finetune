"""Central paths for the Colab urban MMAudio fine-tuning workflow.

The notebooks run from a cloned repo at /content/MMAudio and keep large files
in Google Drive. Keep Drive path changes here so notebooks and helper scripts
do not drift.
"""

from pathlib import Path

REPO_ROOT = Path("/content/MMAudio")
DRIVE_ROOT = Path("/content/drive/MyDrive/MMAudio_Yuqi")

VIDEO_DIR_STRIDE4 = DRIVE_ROOT / "data" / "mmaudio_dataset_stride4"
VIDEO_DIR_STRIDE8 = DRIVE_ROOT / "data" / "mmaudio_dataset_stride8"

LATENT_DIR_STRIDE4 = DRIVE_ROOT / "latents" / "stride4"
LATENT_DIR_STRIDE8 = DRIVE_ROOT / "latents" / "stride8"

WEIGHTS_DIR = DRIVE_ROOT / "weights"
EXT_WEIGHTS_DIR = DRIVE_ROOT / "ext_weights"
OUTPUT_DIR = DRIVE_ROOT / "outputs"
EVAL_DIR = DRIVE_ROOT / "eval_outputs"
LOG_DIR = DRIVE_ROOT / "logs"
TSV_DIR = DRIVE_ROOT / "tsv"

MODE = "16k"
MODEL = "small_16k"
AUDIO_SR = 16000
LATENT_SEQ_LEN = 250
CLIP_SEQ_LEN = 64
SYNC_SEQ_LEN = 192
TEXT_SEQ_LEN = 77

DEFAULT_STRIDE = 8
TEXT_LABEL = "urban soundscape"
VIDEO_EXTENSIONS = (".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v")


def video_dir_for_stride(stride: int = DEFAULT_STRIDE) -> Path:
    if stride == 4:
        return VIDEO_DIR_STRIDE4
    if stride == 8:
        return VIDEO_DIR_STRIDE8
    raise ValueError(f"Unsupported stride: {stride}")


def latent_dir_for_stride(stride: int = DEFAULT_STRIDE) -> Path:
    if stride == 4:
        return LATENT_DIR_STRIDE4
    if stride == 8:
        return LATENT_DIR_STRIDE8
    raise ValueError(f"Unsupported stride: {stride}")


def split_video_dirs(stride: int = DEFAULT_STRIDE) -> dict[str, Path]:
    video_root = video_dir_for_stride(stride)
    return {
        "train": video_root / "train",
        "val": video_root / "val",
        "test": video_root / "test",
    }


def ensure_drive_dirs(stride: int = DEFAULT_STRIDE) -> None:
    for path in [
        DRIVE_ROOT,
        video_dir_for_stride(stride),
        latent_dir_for_stride(stride),
        WEIGHTS_DIR,
        EXT_WEIGHTS_DIR,
        OUTPUT_DIR,
        EVAL_DIR,
        LOG_DIR,
        TSV_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)

