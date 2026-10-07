"""Fail-closed provenance and data contracts for the new official-backed workflow.

Only standard-library imports: validating data must not initialise CUDA or download
weights. These checks do not certify perceptual quality or human review accuracy.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

PINNED_COMMIT = "974010a026c731054592d8f777218bd9d85a6c24"
SPLITS = ("train", "val", "test")
MODEL_SPECS = {
    "small_16k": dict(mode="16k", sample_rate=16000, audio_samples=128000,
                      latent_seq_len=250, latent_dim=20,
                      vae_filename="v1-16.pth", vocoder_filename="best_netG.pt"),
    "small_44k": dict(mode="44k", sample_rate=44100, audio_samples=353280,
                      latent_seq_len=345, latent_dim=40,
                      vae_filename="v1-44.pth", vocoder_filename=None),
}
NORMALIZE_AUDIO = {"train": True, "val": False, "test": False}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, encoding="utf-8",
                                     delete=False) as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def preprocessing_id(model: str) -> str:
    if model not in MODEL_SPECS:
        raise ValueError(f"Unsupported model: {model}")
    payload = {"commit": PINNED_COMMIT, "model": model,
               "normalize_audio": NORMALIZE_AUDIO, "duration_sec": 8.0,
               "dataset": "mmaudio.data.extraction.vgg_sound.VGGSound.sample"}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return "official-v1-" + digest[:24]


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def verify_official_repo(path: str | Path) -> dict:
    repo = Path(path).resolve(strict=True)
    commit = _git(repo, "rev-parse", "HEAD")
    if commit != PINNED_COMMIT:
        raise ValueError(f"Official checkout must be {PINNED_COMMIT}, got {commit}")
    # Include all tracked implementation/configuration files, not only the loader.
    tracked = _git(repo, "ls-files", "-z").split("\0")
    paths = sorted(p for p in tracked if p and (
        p.endswith((".py", ".yaml", ".yml", ".toml"))
        or p in {"requirements.txt", "setup.py"}))
    dirty = _git(repo, "diff", "HEAD", "--name-only", "--", *paths)
    if dirty:
        raise ValueError(f"Official checkout has modified tracked code/config: {dirty}")
    untracked = _git(repo, "ls-files", "--others", "--exclude-standard", "-z")
    unsafe = [p for p in untracked.split("\0") if
              p.startswith(("mmaudio/", "config/", "configs/"))
              and p.endswith((".py", ".yaml", ".yml"))]
    if unsafe:
        raise ValueError(f"Unexpected code/config in official checkout: {unsafe}")
    required = "mmaudio/data/extraction/vgg_sound.py"
    if required not in paths:
        raise ValueError("Official VGGSound implementation missing")
    return {"commit": commit, "files": {p: sha256_file(repo / p) for p in paths}}


def activate_official_repo(path: str | Path) -> dict:
    repo = Path(path).resolve(strict=True)
    provenance = verify_official_repo(repo)
    for name, module in list(sys.modules.items()):
        if name == "mmaudio" or name.startswith("mmaudio."):
            origin = getattr(module, "__file__", None)
            if origin is not None and not Path(origin).resolve().is_relative_to(repo):
                raise ValueError(f"{name} already imported from non-official checkout: {origin}; "
                                 "start a fresh Python process")
    sys.path.insert(0, str(repo))
    return provenance


def read_csv(path: str | Path, delimiter: str = ",") -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def checked_path(path: str, expected_hash: str) -> Path:
    result = Path(path)
    if not result.is_absolute() or not result.is_file():
        raise ValueError(f"Missing absolute file path: {path}")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_hash or ""):
        raise ValueError(f"Missing/invalid SHA256 for {path}")
    if sha256_file(result) != expected_hash:
        raise ValueError(f"SHA256 mismatch: {path}")
    return result


def validate_manifest_rows(rows: list[dict], split_policy: str) -> None:
    if split_policy not in {"source", "site"}:
        raise ValueError("split_policy must be source or site")
    ids: set[str] = set()
    paths: set[str] = set()
    hashes: dict[str, str] = {}
    groups: dict[tuple[str, str], str] = {}
    counts = {s: 0 for s in SPLITS}
    for row in rows:
        identifier = row.get("clip_id", "")
        if not SAFE_ID.fullmatch(identifier) or identifier in ids:
            raise ValueError(f"Unsafe or duplicate clip ID: {identifier!r}")
        ids.add(identifier)
        split = row.get("split")
        if split not in SPLITS:
            raise ValueError(f"Invalid split for {identifier}: {split}")
        counts[split] += 1
        if row.get("manual_review_status") != "included" or row.get("qc_status") != "pass":
            raise ValueError(f"Human review and technical QC must pass: {identifier}")
        if not row.get("label", "").strip() or any(c in row["label"] for c in "\t\r\n"):
            raise ValueError(f"Missing/invalid label: {identifier}")
        path = Path(row.get("path", ""))
        if not path.is_absolute() or path.suffix.lower() != ".mp4":
            raise ValueError(f"Expected absolute .mp4: {path}")
        real_path = str(path.resolve())
        if real_path in paths:
            raise ValueError(f"Same media file assigned multiple clip IDs: {path}")
        paths.add(real_path)
        digest = row.get("sha256", "")
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError(f"Missing media hash: {identifier}")
        if digest in hashes:
            raise ValueError(f"Identical media content for {identifier} and {hashes[digest]}")
        hashes[digest] = identifier
        source = row.get("source_recording_id", "").strip()
        if not source:
            raise ValueError(f"Missing source recording identity: {identifier}")
        for kind, group in [("source", source), ("site", row.get("site_id", "").strip())]:
            if kind == "site" and split_policy != "site":
                continue
            if not group:
                raise ValueError(f"Missing {kind} group: {identifier}")
            key = (kind, group)
            if key in groups and groups[key] != split:
                raise ValueError(f"{kind} leakage across splits: {group}")
            groups[key] = split
    if any(count == 0 for count in counts.values()):
        raise ValueError(f"Every split must be nonempty: {counts}")


def load_plan(path: str | Path) -> dict:
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema_version") != 1 or plan.get("status") != "READY_FOR_OFFICIAL_EXTRACTION":
        raise ValueError("Not a ready official extraction plan")
    if plan.get("official_commit") != PINNED_COMMIT:
        raise ValueError("Plan official commit mismatch")
    model = plan.get("model")
    if model not in MODEL_SPECS or plan.get("preprocessing_id") != preprocessing_id(model):
        raise ValueError("Model/preprocessing contract mismatch")
    if plan.get("normalize_audio") != NORMALIZE_AUDIO:
        raise ValueError("Audio normalization must follow pinned official split policy")
    if set(plan.get("splits", {})) != set(SPLITS):
        raise ValueError("Exactly train/val/test splits required")
    if not plan.get("dataset_version", "").strip():
        raise ValueError("Dataset version required")
    review = plan.get("review_provenance", {})
    summary_path = checked_path(review.get("summary", ""), review.get("summary_sha256", ""))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "final" or summary.get("pending_sources", -1) != 0:
        raise ValueError("Source review is incomplete")
    for file_path, digest in review.get("files", {}).items():
        checked_path(file_path, digest)
    reviewed_path = summary_path.parent / "final_clips_manifest.csv"
    kept_path = summary_path.parent / "kept_sources.csv"
    if any(str(p) not in review.get("files", {}) for p in (reviewed_path, kept_path)):
        raise ValueError("Hashed human-review evidence missing")
    reviewed_rows = read_csv(reviewed_path)
    reviewed = {(r.get("training_id") or r.get("clip_id")): r for r in reviewed_rows}
    if len(reviewed) != len(reviewed_rows) or len(reviewed) != summary.get("final_clips"):
        raise ValueError("Human review count/ID mismatch")

    def source_key(r: dict) -> str:
        return (r.get("source_recording_id") or r.get("source_key")
                or f"{r.get('source_dataset_id', '')}::{r.get('source_file', '')}")

    kept = {source_key(r) for r in read_csv(kept_path) if r.get("review_decision") == "keep"}
    if len(kept) != summary.get("kept_sources"):
        raise ValueError("Kept-source count mismatch")
    all_rows = []
    for split in SPLITS:
        item = plan["splits"][split]
        manifest = checked_path(item["manifest"], item["manifest_sha256"])
        tsv = checked_path(item["tsv"], item["tsv_sha256"])
        rows, labels = read_csv(manifest), read_csv(tsv, "\t")
        if len(rows) != item["count"] or any(row["split"] != split for row in rows):
            raise ValueError(f"Count/split mismatch: {split}")
        expected = [{"id": r["clip_id"], "label": r["label"]} for r in rows]
        if labels != expected:
            raise ValueError(f"TSV/manifest row order or labels differ: {split}")
        video_root = Path(item["video_root"])
        if not video_root.is_absolute():
            raise ValueError("video_root must be absolute")
        for row in rows:
            accepted = reviewed.get(row["clip_id"], {})
            if (accepted.get("manual_review_status") != "included"
                    or accepted.get("split") != split
                    or source_key(accepted) != row["source_recording_id"]
                    or row["source_recording_id"] not in kept):
                raise ValueError(f"Clip disagrees with frozen human review: {row['clip_id']}")
            spec = MODEL_SPECS[model]
            sample_rate = int(row.get("qc_sample_rate") or 0)
            sample_count = int(row.get("qc_audio_samples") or 0)
            if (sample_rate <= 0 or sample_count * spec["sample_rate"]
                    < spec["audio_samples"] * sample_rate):
                raise ValueError(f"Insufficient decoded audio length: {row['clip_id']}")
            media = checked_path(row["path"], row["sha256"])
            staged = video_root / (row["clip_id"] + ".mp4")
            if not staged.is_file() or staged.resolve() != media.resolve():
                raise ValueError(f"Staged video/clip identity mismatch: {staged}")
        all_rows.extend(rows)
    validate_manifest_rows(all_rows, plan["split_policy"])
    return plan
