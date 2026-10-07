"""Prepare immutable, human-approved data plans; never start extraction/training.

Run from the project root with ``python -m fine_tune.official_prepare --help``.
Existing clips are read, hashed and linked; original media are never edited.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import shutil

from fine_tune.official_common import (
    MODEL_SPECS, NORMALIZE_AUDIO, PINNED_COMMIT, SPLITS, SAFE_ID,
    atomic_json, load_plan, preprocessing_id, read_csv, sha256_file,
    validate_manifest_rows, verify_official_repo,
)

VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v"}
FIELDS = ["clip_id", "label", "split", "source_recording_id", "site_id", "path",
          "sha256", "manual_review_status", "qc_status", "qc_audio_samples",
          "qc_sample_rate", "source_start_seconds"]


def write_csv(path: Path, rows: list[dict], fields: list[str], delimiter: str = ",") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", delimiter=delimiter)
        writer.writeheader()
        writer.writerows(rows)


def source_id(row: dict) -> str:
    if row.get("source_recording_id", "").strip():
        return row["source_recording_id"].strip()
    if row.get("source_key", "").strip():
        return row["source_key"].strip()
    if not row.get("source_file", "").strip():
        raise ValueError("source_file/source_recording_id required; do not infer from clip filenames")
    return f"{row.get('source_dataset_id', '')}::{row['source_file']}"


def training_id(row: dict) -> str:
    result = (row.get("training_id") or row.get("clip_id") or "").strip()
    if not SAFE_ID.fullmatch(result):
        raise ValueError(f"Missing/unsafe stable training_id: {result!r}")
    return result


def inventory(source_roots: list[Path], output: Path) -> dict:
    if output.exists():
        raise ValueError(f"Output already exists; choose a new review version: {output}")
    paths = sorted({p.resolve() for root in source_roots for p in root.rglob("*")
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS})
    if not paths:
        raise ValueError("No source videos found; mount the external drive first")
    if len({p.name for p in paths}) != len(paths):
        raise ValueError("Duplicate source filenames across roots: review separate dataset batches")
    output.mkdir(parents=True)
    rows = [{"Source File": p.name, "Source Path": str(p), "Split": "", "Music": "",
             "Decision": "Pending", "Keep From (s)": "", "Notes": "", "site_id": ""}
            for p in paths]
    write_csv(output / "source_review.csv", rows, list(rows[0]))
    result = {"status": "WAITING_FOR_MANUAL_REVIEW", "sources": len(rows),
              "source_roots": [str(p.resolve()) for p in source_roots],
              "review_file": str(output.resolve() / "source_review.csv"),
              "media_files_changed": 0,
              "note": "Human decisions required. Do not change Pending to Keep without review."}
    atomic_json(output / "review_status.json", result)
    return result


def prepare(manifest: Path, review_dir: Path, output: Path, dataset_version: str,
            model: str, split_policy: str = "source", data_root: Path | None = None,
            sync_summary: Path | None = None, default_label: str = "urban soundscape",
            video_layout: str = "existing") -> Path:
    if model not in MODEL_SPECS or not dataset_version.strip():
        raise ValueError("Explicit supported model and dataset_version required")
    if output.exists():
        raise ValueError(f"Refusing to overwrite plan/data: {output}; use a new dataset version")
    summary_path = review_dir / "summary.json"
    review_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if review_summary.get("status") != "final" or review_summary.get("pending_sources", -1) != 0:
        raise ValueError("WAITING_FOR_MANUAL_REVIEW: final source review with zero pending required")
    accepted = read_csv(review_dir / "final_clips_manifest.csv")
    accepted_by_id = {training_id(r): r for r in accepted}
    if not accepted or len(accepted_by_id) != len(accepted):
        raise ValueError("Final reviewed manifest empty or has duplicate training IDs")
    if review_summary.get("final_clips") != len(accepted):
        raise ValueError("Final review summary/manifest count mismatch")
    kept_rows = read_csv(review_dir / "kept_sources.csv")
    kept = {source_id(r) for r in kept_rows if r.get("review_decision") == "keep"}
    if len(kept) != len(kept_rows) or len(kept) != review_summary.get("kept_sources"):
        raise ValueError("Kept-source identities/summary mismatch")
    inputs = read_csv(manifest)
    if not inputs:
        raise ValueError("Empty input clip manifest")
    if any("sync_status" in r for r in inputs):
        if sync_summary is None:
            raise ValueError("Synced inputs require --sync-summary to reject incomplete transfers")
        sync_info = json.loads(sync_summary.read_text(encoding="utf-8"))
        if (sync_info.get("dry_run") is not False or sync_info.get("failed_files") != 0
                or sync_info.get("requested_files") != len(inputs)
                or sync_info.get("synced_files") != len(inputs)
                or sync_info.get("verification") != "sha256"):
            raise ValueError("Drive sync must be complete and SHA256 verified")
    spec = MODEL_SPECS[model]
    result_rows = []
    for row in inputs:
        identifier = training_id(row)
        reviewed = accepted_by_id.get(identifier)
        if reviewed is None:
            raise ValueError(f"Clip not in final human-reviewed manifest: {identifier}")
        if row.get("manual_review_status") != "included" or reviewed.get("manual_review_status") != "included":
            raise ValueError(f"Clip not approved by human review: {identifier}")
        if row.get("qc_status") != "pass":
            raise ValueError(f"Technical QC has not passed: {identifier}")
        if row.get("clapboard_final_status", "included") != "included":
            raise ValueError(f"Clapboard review not final: {identifier}")
        if "sync_status" in row and row["sync_status"] not in {"copied", "skipped", "resumed", "replaced"}:
            raise ValueError(f"Unverified sync entry: {identifier}")
        if source_id(row) != source_id(reviewed) or row.get("split") != reviewed.get("split"):
            raise ValueError(f"Clip source/split changed since human review: {identifier}")
        if source_id(row) not in kept:
            raise ValueError(f"Source was not kept by reviewer: {identifier}")
        reviewed_path = Path(reviewed.get("absolute_path") or "")
        original_path = Path(row.get("original_absolute_path") or row.get("absolute_path") or "")
        if not reviewed_path.is_absolute() or not original_path.is_absolute() or original_path != reviewed_path:
            raise ValueError(f"Media path changed since human review: {identifier}")
        if "sync_status" in row and not re.fullmatch(r"[0-9a-f]{64}", row.get("sync_sha256", "")):
            raise ValueError(f"Missing SHA256 for synced clip: {identifier}")
        if str(row.get("source_start_seconds", "")) != str(reviewed.get("source_start_seconds", "")):
            raise ValueError(f"Clip start changed since human review: {identifier}")
        if str(row.get("qc_audio_length_checked", "")).lower() not in {"true", "1"}:
            raise ValueError(f"Re-run updated QC with audio decoding (not --skip-audio-decode): {identifier}")
        rate = int(row.get("qc_sample_rate") or 0)
        samples = int(row.get("qc_audio_samples") or 0)
        if rate <= 0 or samples < math.ceil(spec["audio_samples"] * rate / spec["sample_rate"]):
            raise ValueError(f"Audio too short for {model}: {identifier} ({samples} samples at {rate}Hz)")
        if data_root is not None:
            relative = Path(row.get("relative_path") or "")
            if relative.is_absolute() or not row.get("relative_path"):
                raise ValueError("Portable relative_path required with --data-root")
            path = (data_root / relative).resolve()
            if not path.is_relative_to(data_root.resolve()):
                raise ValueError(f"Relative path escapes data root: {relative}")
        else:
            original_path = Path(row.get("absolute_path") or "")
            if not original_path.is_absolute():
                raise ValueError(f"Absolute media path required: {identifier}")
            path = original_path.resolve()
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing/empty reviewed clip: {path}")
        digest = sha256_file(path)
        if row.get("sync_sha256") and digest != row["sync_sha256"]:
            raise ValueError(f"Media changed after verified sync: {path}")
        # Older review files did not hash media. Record current bytes; do not claim
        # this retrospectively proves that humans watched these exact bytes.
        label = (row.get("label") or row.get("caption") or default_label).strip()
        result_rows.append({"clip_id": identifier, "label": label, "split": row["split"],
                            "source_recording_id": source_id(row), "site_id": row.get("site_id", ""),
                            "path": str(path), "sha256": digest,
                            "manual_review_status": "included", "qc_status": "pass",
                            "qc_audio_samples": samples, "qc_sample_rate": rate,
                            "source_start_seconds": row.get("source_start_seconds", "")})
    validate_manifest_rows(result_rows, split_policy)
    if video_layout not in {"existing", "symlink"}:
        raise ValueError("video_layout must be existing or symlink")
    existing_roots = {}
    if video_layout == "existing":
        for split in SPLITS:
            paths = [Path(r["path"]) for r in result_rows if r["split"] == split]
            parents = {p.parent for p in paths}
            if len(parents) != 1 or any(Path(r["path"]).name != r["clip_id"] + ".mp4"
                                       for r in result_rows if r["split"] == split):
                raise ValueError("Existing layout needs one folder per split and training_id.mp4 filenames; "
                                 "use verified sync output, or --video-layout symlink on a local filesystem")
            existing_roots[split] = paths[0].parent
    # Only create the output after every input passes. Never remove original files.
    output = output.resolve()
    output.mkdir(parents=True)
    evidence = output / "review_evidence"
    evidence.mkdir()
    evidence_files = {}
    for name in ("summary.json", "final_clips_manifest.csv", "kept_sources.csv"):
        target = evidence / name
        shutil.copyfile(review_dir / name, target)
        evidence_files[str(target)] = sha256_file(target)
    for name, src in (("input_clips_manifest.csv", manifest), ("sync_summary.json", sync_summary)):
        if src is not None:
            target = evidence / name
            shutil.copyfile(src, target)
            evidence_files[str(target)] = sha256_file(target)
    plan = {"schema_version": 1, "status": "READY_FOR_OFFICIAL_EXTRACTION",
            "dataset_version": dataset_version, "model": model, "official_commit": PINNED_COMMIT,
            "preprocessing_id": preprocessing_id(model), "split_policy": split_policy,
            "normalize_audio": NORMALIZE_AUDIO, "splits": {}, "video_layout": video_layout,
            "review_provenance": {"summary": str(evidence / "summary.json"),
                                  "summary_sha256": sha256_file(evidence / "summary.json"),
                                  "files": evidence_files},
            "selection": {"reviewed_clips": len(accepted), "included_clips": len(result_rows),
                          "omitted_reviewed_ids": sorted(set(accepted_by_id) - {r['clip_id'] for r in result_rows})},
            "media_files_changed": 0,
            "scope": "Technical readiness only; official sample decoding still required."
            }
    for split in SPLITS:
        rows = sorted((r for r in result_rows if r["split"] == split), key=lambda r: r["clip_id"])
        if video_layout == "existing":
            video_root = existing_roots[split]
        else:
            video_root = output / "videos" / split
            video_root.mkdir(parents=True)
            for row in rows:
                (video_root / (row["clip_id"] + ".mp4")).symlink_to(row["path"])
        split_manifest, tsv = output / f"{split}_manifest.csv", output / f"{split}.tsv"
        write_csv(split_manifest, rows, FIELDS)
        write_csv(tsv, [{"id": r["clip_id"], "label": r["label"]} for r in rows], ["id", "label"], "\t")
        plan["splits"][split] = {"count": len(rows), "video_root": str(video_root),
                                 "manifest": str(split_manifest), "tsv": str(tsv),
                                 "manifest_sha256": sha256_file(split_manifest), "tsv_sha256": sha256_file(tsv)}
    # Completion marker last. A partially written directory must not be trained.
    candidate = output / "plan.json"
    atomic_json(candidate, plan)
    load_plan(candidate)
    return candidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inv = sub.add_parser("inventory", help="Create an all-Pending human source review CSV; no media changes")
    inv.add_argument("--source-root", action="append", type=Path, required=True)
    inv.add_argument("--output", type=Path, required=True)
    prep = sub.add_parser("prepare", help="Freeze reviewed/QC-passed clips into an official data plan")
    for name in ("manifest", "review-dir", "output"):
        prep.add_argument("--" + name, type=Path, required=True)
    prep.add_argument("--dataset-version", required=True)
    prep.add_argument("--model", choices=sorted(MODEL_SPECS), required=True)
    prep.add_argument("--split-policy", choices=["source", "site"], default="source")
    prep.add_argument("--data-root", type=Path)
    prep.add_argument("--sync-summary", type=Path)
    prep.add_argument("--default-label", default="urban soundscape")
    prep.add_argument("--video-layout", choices=["existing", "symlink"], default="existing",
                      help="existing is Drive-safe; symlink needs a local filesystem such as /content")
    validate = sub.add_parser("validate", help="Recheck hashes, human review, row order and source leakage")
    validate.add_argument("--plan", type=Path, required=True)
    repo = sub.add_parser("verify-repo", help="Verify clean pinned official source; no imports/GPU/downloads")
    repo.add_argument("--official-repo", type=Path, required=True)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "inventory":
        args["source_roots"] = args.pop("source_root")
        print(json.dumps(inventory(**args), indent=2, ensure_ascii=False))
    elif command == "prepare":
        print(prepare(**args))
    elif command == "validate":
        plan = load_plan(args["plan"])
        print(json.dumps({"status": plan["status"], "model": plan["model"],
                          "counts": {s: plan["splits"][s]["count"] for s in SPLITS}}, indent=2))
    else:
        result = verify_official_repo(args["official_repo"])
        print(json.dumps({"commit": result["commit"], "verified_files": len(result["files"])}))


if __name__ == "__main__":
    main()
