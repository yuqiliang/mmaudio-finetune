#!/usr/bin/env python3
"""Combine independently stored clip datasets through exact path manifests."""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
CLIP_FIELDS = (
    "combined_dataset_id",
    "source_dataset_id",
    "combined_clip_id",
    "clip_id",
    "split",
    "source_file",
    "source_start_seconds",
    "duration_seconds",
    "absolute_path",
    "actual_duration_seconds",
    "byte_size",
    "status",
)
SOURCE_FIELDS = (
    "combined_dataset_id",
    "source_dataset_id",
    "source_key",
    "split",
    "source_file",
    "duration_seconds",
    "expected_clips",
    "width",
    "height",
    "frame_rate",
)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_dataset(root: Path, combined_dataset_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    manifest_path = root / "metadata" / "clips_manifest.csv"
    source_path = root / "metadata" / "source_split.csv"
    if not manifest_path.is_file() or not source_path.is_file():
        raise FileNotFoundError(f"Missing metadata under {root}")

    raw_clips = read_csv(manifest_path)
    dataset_ids = {row["dataset_id"] for row in raw_clips}
    if len(dataset_ids) != 1:
        raise ValueError(f"Expected one dataset_id in {manifest_path}, found {sorted(dataset_ids)}")
    source_dataset_id = dataset_ids.pop()

    clips: list[dict[str, Any]] = []
    for row in raw_clips:
        if row.get("status") != "complete":
            raise ValueError(f"Incomplete clip in {manifest_path}: {row.get('clip_id')}")
        absolute_path = (root / row["relative_path"]).resolve()
        if not absolute_path.is_file():
            raise FileNotFoundError(absolute_path)
        actual_size = absolute_path.stat().st_size
        recorded_size = int(row["byte_size"])
        if actual_size != recorded_size:
            raise ValueError(
                f"Size mismatch for {absolute_path}: manifest={recorded_size}, actual={actual_size}"
            )
        clips.append(
            {
                "combined_dataset_id": combined_dataset_id,
                "source_dataset_id": source_dataset_id,
                "combined_clip_id": f"{source_dataset_id}::{row['clip_id']}",
                "clip_id": row["clip_id"],
                "split": row["split"],
                "source_file": row["source_file"],
                "source_start_seconds": row["source_start_seconds"],
                "duration_seconds": row["duration_seconds"],
                "absolute_path": str(absolute_path),
                "actual_duration_seconds": row["actual_duration_seconds"],
                "byte_size": actual_size,
                "status": "complete",
            }
        )

    sources = []
    for row in read_csv(source_path):
        sources.append(
            {
                "combined_dataset_id": combined_dataset_id,
                "source_dataset_id": source_dataset_id,
                "source_key": f"{source_dataset_id}::{row['source_file']}",
                "split": row["split"],
                "source_file": row["source_file"],
                "duration_seconds": row["duration_seconds"],
                "expected_clips": row["expected_clips"],
                "width": row["width"],
                "height": row["height"],
                "frame_rate": row["frame_rate"],
            }
        )
    return clips, sources


def validate_unique(rows: list[dict[str, Any]], field: str) -> None:
    values = [row[field] for row in rows]
    duplicates = [value for value, count in Counter(values).items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate {field}: {duplicates[:5]}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Combine clip datasets without moving or duplicating media files."
    )
    parser.add_argument("--input-root", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--combined-dataset-id",
        default="soundscape_combined_nonoverlap_8s_v1",
    )
    args = parser.parse_args()

    all_clips: list[dict[str, Any]] = []
    all_sources: list[dict[str, Any]] = []
    for root in args.input_root:
        clips, sources = load_dataset(root.expanduser().resolve(), args.combined_dataset_id)
        all_clips.extend(clips)
        all_sources.extend(sources)

    split_order = {split: index for index, split in enumerate(SPLITS)}
    all_clips.sort(
        key=lambda row: (
            split_order[row["split"]],
            row["source_dataset_id"],
            row["source_file"],
            float(row["source_start_seconds"]),
        )
    )
    all_sources.sort(
        key=lambda row: (
            split_order[row["split"]],
            row["source_dataset_id"],
            row["source_file"],
        )
    )
    validate_unique(all_clips, "combined_clip_id")
    validate_unique(all_clips, "absolute_path")
    validate_unique(all_sources, "source_key")

    output = args.output.expanduser().resolve()
    metadata_dir = output / "metadata"
    write_csv(metadata_dir / "clips_manifest.csv", CLIP_FIELDS, all_clips)
    write_csv(metadata_dir / "source_split.csv", SOURCE_FIELDS, all_sources)

    for split in SPLITS:
        paths = [row["absolute_path"] for row in all_clips if row["split"] == split]
        atomic_write_text(metadata_dir / f"{split}_paths.txt", "\n".join(paths) + "\n")

    clip_counts = Counter(row["split"] for row in all_clips)
    source_counts = Counter(row["split"] for row in all_sources)
    actual_seconds: defaultdict[str, float] = defaultdict(float)
    byte_sizes: defaultdict[str, int] = defaultdict(int)
    for row in all_clips:
        actual_seconds[row["split"]] += float(row["actual_duration_seconds"])
        byte_sizes[row["split"]] += int(row["byte_size"])

    summary = {
        "combined_dataset_id": args.combined_dataset_id,
        "input_roots": [str(path.expanduser().resolve()) for path in args.input_root],
        "storage": "logical_manifest_only",
        "splits": {
            split: {
                "sources": source_counts[split],
                "clips": clip_counts[split],
                "clip_hours": round(actual_seconds[split] / 3600, 4),
                "bytes": byte_sizes[split],
            }
            for split in SPLITS
        },
        "total_sources": len(all_sources),
        "total_clips": len(all_clips),
        "total_clip_hours": round(sum(actual_seconds.values()) / 3600, 4),
        "total_bytes": sum(byte_sizes.values()),
    }
    atomic_write_text(metadata_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    atomic_write_text(
        output / "README.txt",
        (
            "This directory is a logical combined dataset. Media files remain in the input roots.\n"
            "Use metadata/{train,val,test}_paths.txt or metadata/clips_manifest.csv so paths are\n"
            "resolved exactly without enumerating the network volume directories.\n"
        ),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
