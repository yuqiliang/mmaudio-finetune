#!/usr/bin/env python3
"""Apply manual clapboard decisions and rebuild final non-destructive manifests."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
MANUAL_DECISIONS = {"keep_all", "exclude_before_cutoff"}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
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


def validate_manual_cutoff(
    row: dict[str, str], clip_seconds: float
) -> float:
    raw_cutoff = (
        row["manual_keep_start_seconds"].strip()
        or row["recommended_keep_start_seconds"].strip()
    )
    if not raw_cutoff:
        raise ValueError(f"Missing cutoff for {row['source_key']}")
    cutoff = float(raw_cutoff)
    candidate_time = float(row["candidate_time_seconds"])
    if cutoff <= candidate_time:
        raise ValueError(
            f"Cutoff must be after the detected clap candidate for {row['source_key']}"
        )
    quotient = cutoff / clip_seconds
    if not math.isclose(quotient, round(quotient), abs_tol=1e-7):
        raise ValueError(
            f"Cutoff must align to a {clip_seconds:g}-second boundary for {row['source_key']}"
        )
    return cutoff


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument("--filter-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clip-seconds", type=float, default=8.0)
    parser.add_argument(
        "--allow-pending",
        action="store_true",
        help="Build an interim manifest while incomplete manual rows remain included.",
    )
    args = parser.parse_args()

    combined_root = args.combined_root.expanduser().resolve()
    filter_root = (
        args.filter_root.expanduser().resolve()
        if args.filter_root
        else combined_root / "metadata" / "clapboard_filter_v1"
    )
    output = (
        args.output.expanduser().resolve()
        if args.output
        else filter_root / "final"
    )
    cutoff_rows = read_csv(filter_root / "cutoff_manifest.csv")
    review_rows = read_csv(filter_root / "manual_review.csv")
    clip_rows = read_csv(combined_root / "metadata" / "clips_manifest.csv")

    review_by_source: dict[str, dict[str, str]] = {}
    for review_row in review_rows:
        source_key = review_row["source_key"]
        if source_key in review_by_source:
            raise ValueError(f"Duplicate manual review row: {source_key}")
        review_by_source[source_key] = review_row

    cutoff_keys = {row["source_key"] for row in cutoff_rows}
    unknown_keys = sorted(set(review_by_source) - cutoff_keys)
    if unknown_keys:
        raise ValueError(f"Unknown source keys in manual review: {unknown_keys[:5]}")

    resolved_rows: list[dict[str, Any]] = []
    pending_sources: list[str] = []
    effective_cutoffs: dict[str, float | None] = {}
    for cutoff_row in cutoff_rows:
        row = dict(cutoff_row)
        source_key = row["source_key"]
        if row["decision"] == "auto_exclude_before_cutoff":
            effective_decision = "auto_exclude_before_cutoff"
            effective_cutoff = float(row["auto_keep_start_seconds"])
        else:
            review_row = review_by_source.get(source_key)
            if review_row is None:
                raise ValueError(f"Missing manual review row: {source_key}")
            manual_decision = review_row["manual_decision"].strip()
            row["manual_decision"] = manual_decision
            row["manual_keep_start_seconds"] = review_row[
                "manual_keep_start_seconds"
            ].strip()
            row["manual_notes"] = review_row["manual_notes"].strip()
            if not manual_decision:
                pending_sources.append(source_key)
                effective_decision = "pending_review"
                effective_cutoff = None
            elif manual_decision not in MANUAL_DECISIONS:
                raise ValueError(
                    f"Invalid manual_decision={manual_decision!r} for {source_key}; "
                    f"expected one of {sorted(MANUAL_DECISIONS)}"
                )
            elif manual_decision == "keep_all":
                effective_decision = "manual_keep_all"
                effective_cutoff = None
            else:
                effective_decision = "manual_exclude_before_cutoff"
                effective_cutoff = validate_manual_cutoff(review_row, args.clip_seconds)

        row["effective_decision"] = effective_decision
        row["effective_keep_start_seconds"] = (
            effective_cutoff if effective_cutoff is not None else ""
        )
        effective_cutoffs[source_key] = effective_cutoff
        resolved_rows.append(row)

    if pending_sources and not args.allow_pending:
        raise RuntimeError(
            f"{len(pending_sources)} manual reviews are incomplete. "
            "Fill manual_decision or pass --allow-pending for an interim manifest."
        )

    excluded_rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    for clip_row in clip_rows:
        source_key = f"{clip_row['source_dataset_id']}::{clip_row['source_file']}"
        cutoff = effective_cutoffs[source_key]
        is_excluded = cutoff is not None and float(
            clip_row["source_start_seconds"]
        ) < cutoff
        result = dict(clip_row)
        result["clapboard_final_status"] = (
            "excluded_before_cutoff"
            if is_excluded
            else (
                "pending_review"
                if source_key in pending_sources
                else "included"
            )
        )
        result["clapboard_effective_keep_start_seconds"] = (
            cutoff if cutoff is not None else ""
        )
        if is_excluded:
            excluded_rows.append(result)
        else:
            filtered_rows.append(result)

    cutoff_fields = list(cutoff_rows[0].keys()) + [
        "effective_decision",
        "effective_keep_start_seconds",
    ]
    clip_fields = list(clip_rows[0].keys()) + [
        "clapboard_final_status",
        "clapboard_effective_keep_start_seconds",
    ]
    write_csv(output / "final_cutoff_manifest.csv", resolved_rows, cutoff_fields)
    write_csv(output / "final_excluded_clips_manifest.csv", excluded_rows, clip_fields)
    write_csv(output / "final_filtered_clips_manifest.csv", filtered_rows, clip_fields)

    for split in SPLITS:
        paths = [
            row["absolute_path"] for row in filtered_rows if row["split"] == split
        ]
        atomic_write_text(
            output / f"final_{split}_paths.txt",
            "\n".join(paths) + ("\n" if paths else ""),
        )

    excluded_counts = Counter(row["split"] for row in excluded_rows)
    filtered_counts = Counter(row["split"] for row in filtered_rows)
    decision_counts = Counter(row["effective_decision"] for row in resolved_rows)
    summary = {
        "source_videos": len(cutoff_rows),
        "decision_counts": dict(decision_counts),
        "pending_review_sources": len(pending_sources),
        "original_clips": len(clip_rows),
        "excluded_clips": len(excluded_rows),
        "filtered_clips": len(filtered_rows),
        "excluded_by_split": {split: excluded_counts[split] for split in SPLITS},
        "filtered_by_split": {split: filtered_counts[split] for split in SPLITS},
        "media_files_moved_or_deleted": 0,
        "status": "interim" if pending_sources else "final",
    }
    atomic_write_text(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
