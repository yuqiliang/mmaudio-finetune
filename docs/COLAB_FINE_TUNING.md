# Colab Fine-tuning Workflow

This workflow fine-tunes `small_44k` on manually reviewed, non-overlapping
8-second soundscape clips. Source videos and generated clips are never deleted
or modified by the review and quality-control tools.

## Fixed model settings

| Setting | Value |
| --- | --- |
| Model | `small_44k` |
| Mode | `44k` |
| Audio sample rate | 44,100 Hz |
| Latent sequence | 345 |
| CLIP sequence | 64 |
| Sync sequence | 192 |
| Sync input | 25 fps, 200 frames |

## 1. Apply the manual source review

Complete the `Decision` column in the source-review workbook. Supported final
values are `Keep` and `Exclude whole source`. `Keep From (s)` is optional for a
retained source and must be an 8-second boundary; earlier clips are then omitted
from the manifest rather than deleted.

```bash
python3 scripts/apply_source_review.py \
  --review "/path/to/MMAudio_manual_quality_review.xlsx" \
  --combined-root "/Volumes/SSID IVR Study 1/Yuqi/MMAudio Combined Dataset 8s" \
  --output "/Volumes/SSID IVR Study 1/Yuqi/MMAudio Final Review"
```

The command preserves the original source-level train/validation/test split and
creates:

- `kept_sources.csv`
- `excluded_sources.csv`
- `pending_sources.csv`
- `final_clips_manifest.csv`
- split manifests, exact path lists, and aligned training TSVs
- `summary.json`

The command stops when a decision is pending. `--allow-pending` can produce a
safe interim manifest, but pending sources are omitted from that manifest.

## 2. Run technical quality control

This check does not classify music or other content. It checks exact manifest
paths for stream readability, duration, 25 fps, 200 frames, 44.1 kHz mono
audio, severe silence, and severe digital clipping.

```bash
python3 scripts/check_clip_quality.py \
  --manifest "/Volumes/SSID IVR Study 1/Yuqi/MMAudio Final Review/final_clips_manifest.csv" \
  --output "/Volumes/SSID IVR Study 1/Yuqi/MMAudio Final Review/quality_control" \
  --workers 4
```

Review `anomalous_sources.csv` and `anomalous_clips.csv`. Clips that pass are
listed in `qc_passed_clips_manifest.csv`; no media files are changed.

## 3. Copy only accepted clips to Drive

Run the sync command where both the external dataset and a mounted Google Drive
destination are visible.

```bash
python3 scripts/sync_manifest_to_drive.py \
  --manifest "/Volumes/SSID IVR Study 1/Yuqi/MMAudio Final Review/quality_control/qc_passed_clips_manifest.csv" \
  --destination "/path/to/Google Drive/My Drive/MMAudio_Yuqi/data/soundscape_nonoverlap_8s" \
  --verify sha256 \
  --workers 2 \
  --retries 3
```

The sync is restartable. It resumes `.part` files, skips verified destination
files, verifies sizes and SHA256 hashes, retries transient failures, and never
deletes source files. A mismatched completed destination is reported instead of
overwritten unless `--replace-mismatch` is explicitly supplied. Synced
manifests include portable `relative_path` values so Colab can resolve them
under `/content/drive` even when the copy was made from macOS or a server.

## 4. Run Colab

Open `notebooks/colab_finetune.ipynb`. It:

1. mounts Drive and syncs the `colab` Git branch;
2. validates the final synced manifest;
3. downloads and stores the `small_44k`, 44.1 kHz VAE, and Synchformer weights
   in Drive;
4. runs a four-clip extraction check;
5. extracts separate train, validation, and test memmaps;
6. validates TSV-to-memmap row alignment;
7. starts or resumes fine-tuning;
8. generates a bounded checkpoint evaluation sample.

Large files are stored below
`/content/drive/MyDrive/MMAudio_Yuqi`. Set `MMAUDIO_DRIVE_ROOT` before importing
`fine_tune.urban_paths` when the Drive uses another root. Hugging Face and
PyTorch model caches are also redirected into this Drive root.
