#!/usr/bin/env python3
"""Detect likely clapboard transients and build non-destructive clip manifests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


SPLITS = ("train", "val", "test")
SAMPLE_RATE = 16_000
FRAME_SECONDS = 0.020
HOP_SECONDS = 0.005
MIN_CANDIDATE_TIME = 0.25
NMS_SECONDS = 0.35
TOP_CANDIDATES = 5


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


def parse_source_roots(values: list[str]) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected DATASET_ID=/path for --source-root: {value}")
        dataset_id, raw_path = value.split("=", 1)
        dataset_id = dataset_id.strip()
        if not dataset_id or dataset_id in roots:
            raise ValueError(f"Invalid or duplicate source dataset id: {dataset_id!r}")
        root = Path(raw_path).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(root)
        roots[dataset_id] = root
    return roots


def decode_audio(source_path: Path, scan_seconds: float) -> np.ndarray:
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-t",
        f"{scan_seconds:.6f}",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-f",
        "f32le",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, capture_output=True)
    audio = np.frombuffer(result.stdout, dtype="<f4").astype(np.float64)
    if audio.size < SAMPLE_RATE:
        raise RuntimeError(f"Decoded less than one second of audio: {source_path}")
    if not np.isfinite(audio).all():
        raise RuntimeError(f"Decoded audio contains non-finite samples: {source_path}")
    return audio


def framed_rms(signal: np.ndarray, frame_size: int, hop_size: int) -> np.ndarray:
    if signal.size < frame_size:
        return np.empty(0, dtype=np.float64)
    starts = np.arange(0, signal.size - frame_size + 1, hop_size, dtype=np.int64)
    power = signal * signal
    cumulative = np.concatenate(([0.0], np.cumsum(power, dtype=np.float64)))
    sums = cumulative[starts + frame_size] - cumulative[starts]
    return np.sqrt(np.maximum(sums / frame_size, 1e-20))


def robust_scale(values: np.ndarray) -> tuple[float, float]:
    median = float(np.median(values))
    mad = float(1.4826 * np.median(np.abs(values - median)))
    return median, max(mad, 1.0)


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def candidate_metrics(
    index: int,
    transient_db: np.ndarray,
    raw_db: np.ndarray,
    audio: np.ndarray,
    global_median: float,
    global_scale: float,
    hop_size: int,
) -> dict[str, float]:
    time_seconds = index * HOP_SECONDS + FRAME_SECONDS / 2
    local_radius = round(1.0 / HOP_SECONDS)
    exclusion_radius = round(0.080 / HOP_SECONDS)
    left = max(0, index - local_radius)
    right = min(transient_db.size, index + local_radius + 1)
    before = transient_db[left : max(left, index - exclusion_radius)]
    after = transient_db[min(right, index + exclusion_radius + 1) : right]
    baseline_values = np.concatenate((before, after))
    local_baseline = (
        float(np.median(baseline_values)) if baseline_values.size else global_median
    )
    prominence_db = float(transient_db[index] - local_baseline)
    robust_z = float((transient_db[index] - global_median) / global_scale)

    center_sample = round(time_seconds * SAMPLE_RATE)
    peak_radius = round(0.020 * SAMPLE_RATE)
    rms_radius = round(0.250 * SAMPLE_RATE)
    peak_slice = audio[
        max(0, center_sample - peak_radius) : min(audio.size, center_sample + peak_radius + 1)
    ]
    rms_slice = audio[
        max(0, center_sample - rms_radius) : min(audio.size, center_sample + rms_radius + 1)
    ]
    peak_amplitude = float(np.max(np.abs(peak_slice))) if peak_slice.size else 0.0
    local_rms = float(np.sqrt(np.mean(rms_slice * rms_slice))) if rms_slice.size else 0.0
    crest_db = float(20 * math.log10((peak_amplitude + 1e-9) / (local_rms + 1e-9)))
    high_frequency_ratio_db = float(transient_db[index] - raw_db[index])

    half_height = local_baseline + max(prominence_db, 0.0) / 2
    width_left = index
    width_right = index
    max_width_frames = round(0.5 / HOP_SECONDS)
    while (
        width_left > 0
        and index - width_left < max_width_frames
        and transient_db[width_left - 1] >= half_height
    ):
        width_left -= 1
    while (
        width_right + 1 < transient_db.size
        and width_right - index < max_width_frames
        and transient_db[width_right + 1] >= half_height
    ):
        width_right += 1
    width_seconds = (width_right - width_left + 1) * HOP_SECONDS

    early_bonus = 2.0 if time_seconds <= 30.0 else 0.0
    narrow_bonus = 2.0 * clamp01((0.25 - width_seconds) / 0.20)
    rank_score = (
        prominence_db
        + 0.45 * min(robust_z, 25.0)
        + 0.25 * min(crest_db, 30.0)
        + early_bonus
        + narrow_bonus
    )
    return {
        "time_seconds": round(time_seconds, 4),
        "rank_score": round(rank_score, 4),
        "peak_transient_db": round(float(transient_db[index]), 4),
        "local_baseline_db": round(local_baseline, 4),
        "prominence_db": round(prominence_db, 4),
        "robust_z": round(robust_z, 4),
        "crest_db": round(crest_db, 4),
        "high_frequency_ratio_db": round(high_frequency_ratio_db, 4),
        "width_seconds": round(width_seconds, 4),
        "peak_amplitude": round(peak_amplitude, 7),
    }


def analyze_audio(audio: np.ndarray, priority_seconds: float) -> dict[str, Any]:
    frame_size = round(FRAME_SECONDS * SAMPLE_RATE)
    hop_size = round(HOP_SECONDS * SAMPLE_RATE)
    centered = audio - float(np.median(audio))
    preemphasized = np.empty_like(centered)
    preemphasized[0] = centered[0]
    preemphasized[1:] = centered[1:] - 0.97 * centered[:-1]
    transient_rms = framed_rms(preemphasized, frame_size, hop_size)
    raw_rms = framed_rms(centered, frame_size, hop_size)
    transient_db = 20 * np.log10(np.maximum(transient_rms, 1e-9))
    raw_db = 20 * np.log10(np.maximum(raw_rms, 1e-9))
    global_median, global_scale = robust_scale(transient_db)

    local_maxima = np.flatnonzero(
        (transient_db[1:-1] > transient_db[:-2])
        & (transient_db[1:-1] >= transient_db[2:])
    ) + 1
    minimum_index = round(MIN_CANDIDATE_TIME / HOP_SECONDS)
    local_maxima = local_maxima[local_maxima >= minimum_index]
    if not local_maxima.size:
        return {
            "audio_peak_amplitude": round(float(np.max(np.abs(audio))), 7),
            "transient_median_db": round(global_median, 4),
            "transient_robust_scale_db": round(global_scale, 4),
            "candidates": [],
        }

    strongest_indices = local_maxima[np.argsort(transient_db[local_maxima])[::-1][:400]]
    measured = [
        candidate_metrics(
            int(index),
            transient_db,
            raw_db,
            centered,
            global_median,
            global_scale,
            hop_size,
        )
        for index in strongest_indices
    ]
    measured.sort(key=lambda item: item["rank_score"], reverse=True)

    selected: list[dict[str, float]] = []
    for candidate in measured:
        if any(
            abs(candidate["time_seconds"] - existing["time_seconds"]) < NMS_SECONDS
            for existing in selected
        ):
            continue
        selected.append(candidate)
        if len(selected) == TOP_CANDIDATES:
            break

    for candidate in selected:
        candidate["in_priority_window"] = float(
            candidate["time_seconds"] <= priority_seconds
        )
    return {
        "audio_peak_amplitude": round(float(np.max(np.abs(audio))), 7),
        "transient_median_db": round(global_median, 4),
        "transient_robust_scale_db": round(global_scale, 4),
        "candidates": selected,
    }


def confidence_for_candidates(
    candidates: list[dict[str, float]],
    priority_seconds: float,
    auto_min_time: float,
    min_rank_margin: float,
) -> tuple[str, float, str]:
    if not candidates:
        return "no_peak", 0.0, "no_local_peak"
    top = candidates[0]
    margin = (
        top["rank_score"] - candidates[1]["rank_score"]
        if len(candidates) > 1
        else top["rank_score"]
    )
    prominence_score = clamp01((top["prominence_db"] - 7.0) / 15.0)
    robust_score = clamp01((top["robust_z"] - 3.0) / 10.0)
    crest_score = clamp01((top["crest_db"] - 6.0) / 16.0)
    width_score = clamp01((0.24 - top["width_seconds"]) / 0.20)
    margin_score = clamp01(margin / 8.0)
    early_score = 1.0 if top["time_seconds"] <= priority_seconds else 0.45
    confidence = (
        0.30 * prominence_score
        + 0.24 * robust_score
        + 0.14 * crest_score
        + 0.12 * width_score
        + 0.12 * margin_score
        + 0.08 * early_score
    )
    confidence = round(confidence, 4)

    obvious_peak = top["prominence_db"] >= 7.0 and top["robust_z"] >= 3.5
    high_shape = (
        top["prominence_db"] >= 11.0
        and top["robust_z"] >= 5.5
        and top["crest_db"] >= 8.0
        and top["width_seconds"] <= 0.20
    )
    high_timing = auto_min_time <= top["time_seconds"] <= priority_seconds
    high_uniqueness = margin >= min_rank_margin
    if high_shape and high_timing and high_uniqueness and confidence >= 0.75:
        return "high", confidence, "strong_unique_transient"
    if high_shape and not high_timing:
        return "medium", confidence, "candidate_outside_auto_time_window"
    if high_shape and not high_uniqueness:
        return "medium", confidence, "multiple_competing_transients"
    if obvious_peak and confidence >= 0.48:
        return "medium", confidence, "plausible_transient_needs_review"
    return "low", confidence, "weak_or_ambiguous_transient"


def strict_next_boundary(time_seconds: float, boundary_seconds: float) -> float:
    return (math.floor(time_seconds / boundary_seconds) + 1) * boundary_seconds


def state_name(source_key: str) -> str:
    digest = hashlib.sha1(source_key.encode("utf-8")).hexdigest()[:12]
    return f"{digest}.json"


def load_or_analyze_source(
    source_row: dict[str, str],
    source_path: Path,
    state_dir: Path,
    scan_seconds: float,
    priority_seconds: float,
) -> dict[str, Any]:
    source_stat = source_path.stat()
    state_path = state_dir / state_name(source_row["source_key"])
    fingerprint = {
        "source_key": source_row["source_key"],
        "source_path": str(source_path),
        "source_size": source_stat.st_size,
        "source_mtime_ns": source_stat.st_mtime_ns,
        "scan_seconds": scan_seconds,
        "priority_seconds": priority_seconds,
        "sample_rate": SAMPLE_RATE,
        "frame_seconds": FRAME_SECONDS,
        "hop_seconds": HOP_SECONDS,
    }
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state.get("fingerprint") == fingerprint:
            return state

    effective_scan = min(scan_seconds, float(source_row["duration_seconds"]))
    audio = decode_audio(source_path, effective_scan)
    analysis = analyze_audio(audio, priority_seconds)
    source_stat_after = source_path.stat()
    if (
        source_stat.st_size,
        source_stat.st_mtime_ns,
    ) != (
        source_stat_after.st_size,
        source_stat_after.st_mtime_ns,
    ):
        raise RuntimeError(f"Source changed during analysis: {source_path}")
    state = {
        "fingerprint": fingerprint,
        "effective_scan_seconds": round(audio.size / SAMPLE_RATE, 4),
        "analysis": analysis,
    }
    atomic_write_text(state_path, json.dumps(state, indent=2, sort_keys=True) + "\n")
    return state


def make_review_audio(
    source_path: Path,
    output_path: Path,
    candidate_time: float | None,
    snippet_seconds: float,
) -> tuple[float, float]:
    if candidate_time is None:
        start_seconds = 0.0
    else:
        start_seconds = max(0.0, candidate_time - snippet_seconds / 2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_seconds:.4f}",
        "-t",
        f"{snippet_seconds:.4f}",
        "-i",
        str(source_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(SAMPLE_RATE),
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    candidate_offset = (
        round(candidate_time - start_seconds, 4) if candidate_time is not None else 0.0
    )
    return round(start_seconds, 4), candidate_offset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        required=True,
        help="Map a source dataset id to its read-only long-video root: DATASET_ID=/path",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--scan-seconds", type=float, default=60.0)
    parser.add_argument("--priority-seconds", type=float, default=30.0)
    parser.add_argument("--auto-min-time", type=float, default=0.75)
    parser.add_argument("--min-rank-margin", type=float, default=6.0)
    parser.add_argument("--clip-seconds", type=float, default=8.0)
    parser.add_argument("--snippet-seconds", type=float, default=10.0)
    parser.add_argument("--skip-review-audio", action="store_true")
    args = parser.parse_args()

    combined_root = args.combined_root.expanduser().resolve()
    metadata_dir = combined_root / "metadata"
    source_rows = read_csv(metadata_dir / "source_split.csv")
    clip_rows = read_csv(metadata_dir / "clips_manifest.csv")
    source_roots = parse_source_roots(args.source_root)
    output = (
        args.output.expanduser().resolve()
        if args.output
        else metadata_dir / "clapboard_filter_v1"
    )
    state_dir = output / "state"
    state_dir.mkdir(parents=True, exist_ok=True)

    missing_roots = sorted(
        {row["source_dataset_id"] for row in source_rows} - set(source_roots)
    )
    if missing_roots:
        raise ValueError(f"Missing --source-root mappings: {missing_roots}")

    cutoff_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    source_paths: dict[str, Path] = {}
    for index, source_row in enumerate(source_rows, start=1):
        source_key = source_row["source_key"]
        source_path = (
            source_roots[source_row["source_dataset_id"]] / source_row["source_file"]
        ).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        source_paths[source_key] = source_path
        print(f"[{index:02d}/{len(source_rows)}] {source_row['source_file']}", flush=True)
        state = load_or_analyze_source(
            source_row,
            source_path,
            state_dir,
            args.scan_seconds,
            args.priority_seconds,
        )
        candidates = state["analysis"]["candidates"]
        confidence_band, confidence_score, reason = confidence_for_candidates(
            candidates,
            args.priority_seconds,
            args.auto_min_time,
            args.min_rank_margin,
        )
        top = candidates[0] if candidates else None
        second = candidates[1] if len(candidates) > 1 else None
        candidate_time = float(top["time_seconds"]) if top else None
        recommended_keep_start = (
            strict_next_boundary(candidate_time, args.clip_seconds)
            if candidate_time is not None
            else None
        )
        decision = "auto_exclude_before_cutoff" if confidence_band == "high" else "pending_review"
        auto_keep_start = recommended_keep_start if confidence_band == "high" else None
        snippet_name = (
            f"{hashlib.sha1(source_key.encode('utf-8')).hexdigest()[:10]}_"
            f"{Path(source_row['source_file']).stem}.m4a"
        )
        snippet_relative_path = f"review_audio/{snippet_name}"
        snippet_start = ""
        candidate_offset = ""
        if not args.skip_review_audio:
            snippet_start, candidate_offset = make_review_audio(
                source_path,
                output / snippet_relative_path,
                candidate_time,
                args.snippet_seconds,
            )

        cutoff_rows.append(
            {
                "combined_dataset_id": source_row["combined_dataset_id"],
                "source_dataset_id": source_row["source_dataset_id"],
                "source_key": source_key,
                "split": source_row["split"],
                "source_file": source_row["source_file"],
                "source_path": str(source_path),
                "scan_seconds": state["effective_scan_seconds"],
                "priority_seconds": args.priority_seconds,
                "auto_min_time_seconds": args.auto_min_time,
                "minimum_rank_margin": args.min_rank_margin,
                "confidence_band": confidence_band,
                "confidence_score": confidence_score,
                "decision": decision,
                "decision_reason": reason,
                "candidate_time_seconds": candidate_time if candidate_time is not None else "",
                "recommended_keep_start_seconds": (
                    recommended_keep_start if recommended_keep_start is not None else ""
                ),
                "auto_keep_start_seconds": auto_keep_start if auto_keep_start is not None else "",
                "candidate_rank_score": top["rank_score"] if top else "",
                "candidate_prominence_db": top["prominence_db"] if top else "",
                "candidate_robust_z": top["robust_z"] if top else "",
                "candidate_crest_db": top["crest_db"] if top else "",
                "candidate_width_seconds": top["width_seconds"] if top else "",
                "second_candidate_time_seconds": second["time_seconds"] if second else "",
                "second_candidate_rank_score": second["rank_score"] if second else "",
                "review_snippet": snippet_relative_path,
                "snippet_start_seconds": snippet_start,
                "candidate_offset_in_snippet_seconds": candidate_offset,
                "manual_decision": "",
                "manual_keep_start_seconds": "",
                "manual_notes": "",
            }
        )
        for rank, candidate in enumerate(candidates, start=1):
            candidate_rows.append(
                {
                    "source_dataset_id": source_row["source_dataset_id"],
                    "source_key": source_key,
                    "split": source_row["split"],
                    "source_file": source_row["source_file"],
                    "candidate_rank": rank,
                    **candidate,
                }
            )

    cutoff_by_source = {row["source_key"]: row for row in cutoff_rows}
    excluded_rows: list[dict[str, Any]] = []
    filtered_rows: list[dict[str, Any]] = []
    appended_fields = [
        "clapboard_filter_status",
        "clapboard_candidate_time_seconds",
        "clapboard_keep_start_seconds",
    ]
    for clip_row in clip_rows:
        source_key = f"{clip_row['source_dataset_id']}::{clip_row['source_file']}"
        cutoff = cutoff_by_source[source_key]
        keep_start_raw = cutoff["auto_keep_start_seconds"]
        is_excluded = bool(keep_start_raw) and float(
            clip_row["source_start_seconds"]
        ) < float(keep_start_raw)
        result = dict(clip_row)
        result.update(
            {
                "clapboard_filter_status": (
                    "auto_excluded"
                    if is_excluded
                    else (
                        "pending_review"
                        if cutoff["decision"] == "pending_review"
                        else "included_after_cutoff"
                    )
                ),
                "clapboard_candidate_time_seconds": cutoff["candidate_time_seconds"],
                "clapboard_keep_start_seconds": keep_start_raw,
            }
        )
        if is_excluded:
            excluded_rows.append(result)
        else:
            filtered_rows.append(result)

    clip_fields = list(clip_rows[0].keys()) + appended_fields
    cutoff_fields = list(cutoff_rows[0].keys())
    candidate_fields = list(candidate_rows[0].keys())
    review_rows = [row for row in cutoff_rows if row["decision"] == "pending_review"]
    write_csv(output / "cutoff_manifest.csv", cutoff_rows, cutoff_fields)
    write_csv(output / "manual_review.csv", review_rows, cutoff_fields)
    write_csv(output / "clapboard_candidates.csv", candidate_rows, candidate_fields)
    write_csv(output / "excluded_clips_manifest.csv", excluded_rows, clip_fields)
    write_csv(output / "filtered_clips_manifest.csv", filtered_rows, clip_fields)

    review_playlist = ["#EXTM3U"]
    auto_playlist = ["#EXTM3U"]
    for row in cutoff_rows:
        entry = [
            (
                f"#EXTINF:{args.snippet_seconds:g},{row['source_file']} | "
                f"candidate {row['candidate_time_seconds']}s | {row['confidence_band']}"
            ),
            row["review_snippet"],
        ]
        if row["decision"] == "pending_review":
            review_playlist.extend(entry)
        else:
            auto_playlist.extend(entry)
    atomic_write_text(
        output / "manual_review_playlist.m3u8",
        "\n".join(review_playlist) + "\n",
    )
    atomic_write_text(
        output / "auto_cutoff_audit_playlist.m3u8",
        "\n".join(auto_playlist) + "\n",
    )

    for split in SPLITS:
        paths = [
            row["absolute_path"] for row in filtered_rows if row["split"] == split
        ]
        atomic_write_text(
            output / f"filtered_{split}_paths.txt",
            "\n".join(paths) + ("\n" if paths else ""),
        )

    confidence_counts = Counter(row["confidence_band"] for row in cutoff_rows)
    excluded_counts = Counter(row["split"] for row in excluded_rows)
    filtered_counts = Counter(row["split"] for row in filtered_rows)
    summary = {
        "method": "audio_transient_scan_v1",
        "source_videos": len(source_rows),
        "scan_seconds": args.scan_seconds,
        "priority_seconds": args.priority_seconds,
        "auto_min_time_seconds": args.auto_min_time,
        "minimum_rank_margin": args.min_rank_margin,
        "clip_boundary_seconds": args.clip_seconds,
        "confidence_counts": dict(confidence_counts),
        "auto_cutoff_sources": sum(
            row["decision"] == "auto_exclude_before_cutoff" for row in cutoff_rows
        ),
        "pending_review_sources": len(review_rows),
        "original_clips": len(clip_rows),
        "auto_excluded_clips": len(excluded_rows),
        "filtered_clips_before_manual_review": len(filtered_rows),
        "excluded_by_split": {split: excluded_counts[split] for split in SPLITS},
        "filtered_by_split": {split: filtered_counts[split] for split in SPLITS},
        "media_files_moved_or_deleted": 0,
        "warning": (
            "Pending-review sources remain included. Apply manual decisions before training."
        ),
    }
    atomic_write_text(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    atomic_write_text(
        output / "README.txt",
        (
            "Clapboard filtering is manifest-only. No media files were moved or deleted.\n"
            "High-confidence sources are filtered automatically at the next strict 8-second boundary.\n"
            "Pending-review sources remain included until manual_review.csv is completed and applied.\n"
            "Review snippets are 10-second mono previews centered on the strongest candidate when possible.\n"
            "For each manual row, set manual_decision to keep_all or exclude_before_cutoff.\n"
            "For exclude_before_cutoff, leave manual_keep_start_seconds blank to accept the recommended\n"
            "8-second boundary, or enter another aligned boundary after the confirmed clap.\n"
            "Run scripts/apply_clapboard_review.py after all manual decisions are complete.\n"
        ),
    )
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
