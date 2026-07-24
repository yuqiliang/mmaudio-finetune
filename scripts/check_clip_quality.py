#!/usr/bin/env python3
"""Run read-only technical checks on manifest-listed clips."""

from __future__ import annotations

import argparse
import array
import csv
import json
import math
import os
import subprocess
import tempfile
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class QualityConfig:
    expected_duration: float = 8.0
    duration_tolerance: float = 0.15
    expected_fps: float = 25.0
    fps_tolerance: float = 0.01
    expected_frames: int = 200
    expected_sample_rate: int = 44100
    expected_channels: int = 1
    silence_rms_db: float = -55.0
    silence_peak_db: float = -40.0
    clipping_amplitude: float = 0.999
    clipping_ratio: float = 0.001
    decode_audio: bool = True


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fieldnames: list[str],
    *,
    delimiter: str = ",",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            delimiter=delimiter,
        )
        writer.writeheader()
        writer.writerows(rows)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


def union_fields(rows: list[dict[str, Any]], required: list[str]) -> list[str]:
    fields = list(required)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def rate(value: Any) -> float:
    text = str(value or "0")
    try:
        return float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        return 0.0


def dbfs(amplitude: float) -> float:
    return 20.0 * math.log10(max(amplitude, 1e-12))


def probe(path: Path) -> tuple[dict[str, Any] | None, str]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-show_entries",
        (
            "format=duration:"
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,"
            "avg_frame_rate,nb_read_frames,nb_frames,sample_rate,channels"
        ),
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        return None, result.stderr.strip() or f"ffprobe exited {result.returncode}"
    try:
        return json.loads(result.stdout), result.stderr.strip()
    except json.JSONDecodeError as error:
        return None, f"invalid ffprobe JSON: {error}"


def analyze_audio(path: Path, sample_rate: int, config: QualityConfig) -> dict[str, Any]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-nostdin",
        "-i",
        str(path),
        "-map",
        "0:a:0",
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-f",
        "f32le",
        "pipe:1",
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    sample_count = 0
    sum_squares = 0.0
    peak = 0.0
    clipped = 0
    remainder = b""
    while True:
        chunk = process.stdout.read(1024 * 1024)
        if not chunk:
            break
        payload = remainder + chunk
        usable = len(payload) - (len(payload) % 4)
        remainder = payload[usable:]
        samples = array.array("f")
        samples.frombytes(payload[:usable])
        if os.sys.byteorder != "little":
            samples.byteswap()
        for value in samples:
            absolute = abs(value)
            sample_count += 1
            sum_squares += value * value
            peak = max(peak, absolute)
            clipped += int(absolute >= config.clipping_amplitude)
    assert process.stderr is not None
    stderr = process.stderr.read().decode("utf-8", errors="replace").strip()
    return_code = process.wait()
    if return_code != 0:
        raise RuntimeError(stderr or f"ffmpeg audio decode exited {return_code}")
    if remainder:
        raise RuntimeError("ffmpeg returned an incomplete float32 sample")
    if sample_count == 0:
        raise RuntimeError("decoded audio contains no samples")
    rms = math.sqrt(sum_squares / sample_count)
    return {
        "qc_audio_samples": sample_count,
        "qc_audio_rms_db": round(dbfs(rms), 4),
        "qc_audio_peak_db": round(dbfs(peak), 4),
        "qc_clipped_samples": clipped,
        "qc_clipping_ratio": round(clipped / sample_count, 8),
    }


def check_row(row: dict[str, str], config: QualityConfig) -> dict[str, Any]:
    result: dict[str, Any] = dict(row)
    reasons: list[str] = []
    path = Path(row["absolute_path"])
    result.update(
        {
            "qc_file_exists": path.is_file(),
            "qc_duration_seconds": "",
            "qc_fps": "",
            "qc_frame_count": "",
            "qc_sample_rate": "",
            "qc_channels": "",
            "qc_audio_samples": "",
            "qc_audio_rms_db": "",
            "qc_audio_peak_db": "",
            "qc_clipped_samples": "",
            "qc_clipping_ratio": "",
        }
    )
    if not path.is_file():
        reasons.append("missing_file")
    else:
        actual_size = path.stat().st_size
        result["qc_byte_size"] = actual_size
        recorded_size = row.get("byte_size", "").strip()
        if recorded_size and actual_size != int(recorded_size):
            reasons.append("byte_size_mismatch")

    payload = None
    if not reasons or reasons == ["byte_size_mismatch"]:
        payload, probe_error = probe(path)
        if payload is None:
            reasons.append("ffprobe_failed")
            result["qc_probe_error"] = probe_error
        elif probe_error:
            result["qc_probe_warning"] = probe_error

    if payload is not None:
        streams = payload.get("streams", [])
        video = next(
            (stream for stream in streams if stream.get("codec_type") == "video"),
            None,
        )
        audio = next(
            (stream for stream in streams if stream.get("codec_type") == "audio"),
            None,
        )
        try:
            duration = float(payload.get("format", {}).get("duration", 0))
        except (TypeError, ValueError):
            duration = 0.0
        result["qc_duration_seconds"] = round(duration, 6)
        if abs(duration - config.expected_duration) > config.duration_tolerance:
            reasons.append("duration_out_of_range")

        if video is None:
            reasons.append("missing_video_stream")
        else:
            fps = rate(video.get("avg_frame_rate") or video.get("r_frame_rate"))
            result["qc_fps"] = round(fps, 6)
            if abs(fps - config.expected_fps) > config.fps_tolerance:
                reasons.append("unexpected_fps")
            raw_frames = video.get("nb_read_frames") or video.get("nb_frames")
            try:
                frame_count = int(raw_frames)
            except (TypeError, ValueError):
                frame_count = round(duration * fps)
                result["qc_frame_count_estimated"] = True
            result["qc_frame_count"] = frame_count
            if frame_count != config.expected_frames:
                reasons.append("unexpected_frame_count")

        if audio is None:
            reasons.append("missing_audio_stream")
        else:
            try:
                sample_rate = int(audio.get("sample_rate", 0))
            except (TypeError, ValueError):
                sample_rate = 0
            try:
                channels = int(audio.get("channels", 0))
            except (TypeError, ValueError):
                channels = 0
            result["qc_sample_rate"] = sample_rate
            result["qc_channels"] = channels
            if sample_rate != config.expected_sample_rate:
                reasons.append("unexpected_sample_rate")
            if channels != config.expected_channels:
                reasons.append("unexpected_channels")

            if config.decode_audio:
                try:
                    audio_metrics = analyze_audio(
                        path,
                        config.expected_sample_rate,
                        config,
                    )
                    result.update(audio_metrics)
                    if (
                        audio_metrics["qc_audio_rms_db"] <= config.silence_rms_db
                        or audio_metrics["qc_audio_peak_db"] <= config.silence_peak_db
                    ):
                        reasons.append("severe_silence")
                    if audio_metrics["qc_clipping_ratio"] > config.clipping_ratio:
                        reasons.append("severe_clipping")
                except Exception as error:
                    reasons.append("audio_decode_failed")
                    result["qc_audio_error"] = str(error)

    result["qc_status"] = "pass" if not reasons else "fail"
    result["qc_reasons"] = ";".join(dict.fromkeys(reasons))
    return result


def source_key(row: dict[str, Any]) -> str:
    return (
        row.get("source_key")
        or f"{row.get('source_dataset_id', '')}::{row.get('source_file', '')}"
    )


def write_passed_outputs(
    output: Path,
    passed: list[dict[str, Any]],
    caption: str,
) -> None:
    fields = union_fields(passed, ["training_id", "split", "absolute_path", "qc_status"])
    write_csv(output / "qc_passed_clips_manifest.csv", passed, fields)
    for split in SPLITS:
        split_rows = [row for row in passed if row["split"] == split]
        write_csv(
            output / f"qc_passed_{split}_clips_manifest.csv",
            split_rows,
            fields,
        )
        atomic_write_text(
            output / f"qc_passed_{split}_paths.txt",
            "\n".join(str(row["absolute_path"]) for row in split_rows)
            + ("\n" if split_rows else ""),
        )
        write_csv(
            output / f"video_{split}_ft.tsv",
            [
                {"id": row["training_id"], "caption": caption}
                for row in split_rows
            ],
            ["id", "caption"],
            delimiter="\t",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--caption", default="urban soundscape")
    parser.add_argument("--expected-duration", type=float, default=8.0)
    parser.add_argument("--duration-tolerance", type=float, default=0.15)
    parser.add_argument("--expected-fps", type=float, default=25.0)
    parser.add_argument("--expected-frames", type=int, default=200)
    parser.add_argument("--expected-sample-rate", type=int, default=44100)
    parser.add_argument("--expected-channels", type=int, default=1)
    parser.add_argument("--silence-rms-db", type=float, default=-55.0)
    parser.add_argument("--silence-peak-db", type=float, default=-40.0)
    parser.add_argument("--clipping-ratio", type=float, default=0.001)
    parser.add_argument("--skip-audio-decode", action="store_true")
    parser.add_argument("--fail-on-anomaly", action="store_true")
    args = parser.parse_args()

    config = QualityConfig(
        expected_duration=args.expected_duration,
        duration_tolerance=args.duration_tolerance,
        expected_fps=args.expected_fps,
        expected_frames=args.expected_frames,
        expected_sample_rate=args.expected_sample_rate,
        expected_channels=args.expected_channels,
        silence_rms_db=args.silence_rms_db,
        silence_peak_db=args.silence_peak_db,
        clipping_ratio=args.clipping_ratio,
        decode_audio=not args.skip_audio_decode,
    )
    manifest = args.manifest.expanduser().resolve()
    output = args.output.expanduser().resolve()
    rows = read_csv(manifest)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")
    if "absolute_path" not in rows[0]:
        raise ValueError("Manifest must contain an absolute_path column")
    output.mkdir(parents=True, exist_ok=True)
    progress_lock = threading.Lock()
    completed = 0

    def run(item: tuple[int, dict[str, str]]) -> tuple[int, dict[str, Any]]:
        nonlocal completed
        index, row = item
        checked = check_row(row, config)
        with progress_lock:
            completed += 1
            if completed == len(rows) or completed % 50 == 0:
                print(f"Checked {completed}/{len(rows)} clips")
        return index, checked

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        checked_pairs = list(executor.map(run, enumerate(rows)))
    checked = [row for _, row in sorted(checked_pairs)]
    passed = [row for row in checked if row["qc_status"] == "pass"]
    failed = [row for row in checked if row["qc_status"] == "fail"]

    all_fields = union_fields(
        checked,
        ["training_id", "split", "absolute_path", "qc_status", "qc_reasons"],
    )
    write_csv(output / "clip_quality_manifest.csv", checked, all_fields)
    write_csv(output / "anomalous_clips.csv", failed, all_fields)
    write_passed_outputs(output, passed, args.caption)

    source_failures: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in failed:
        source_failures[source_key(row)].append(row)
    anomalous_sources: list[dict[str, Any]] = []
    for key, source_rows in sorted(source_failures.items()):
        reason_counts = Counter()
        for row in source_rows:
            reason_counts.update(filter(None, row["qc_reasons"].split(";")))
        anomalous_sources.append(
            {
                "source_key": key,
                "source_file": source_rows[0].get("source_file", ""),
                "split": source_rows[0].get("split", ""),
                "failed_clips": len(source_rows),
                "reasons": ";".join(
                    f"{reason}:{count}" for reason, count in sorted(reason_counts.items())
                ),
                "example_clip_ids": ";".join(
                    str(row.get("training_id") or row.get("clip_id"))
                    for row in source_rows[:10]
                ),
            }
        )
    write_csv(
        output / "anomalous_sources.csv",
        anomalous_sources,
        ["source_key", "source_file", "split", "failed_clips", "reasons", "example_clip_ids"],
    )

    reason_counts = Counter()
    split_pass = Counter(row["split"] for row in passed)
    split_fail = Counter(row["split"] for row in failed)
    for row in failed:
        reason_counts.update(filter(None, row["qc_reasons"].split(";")))
    summary = {
        "manifest": str(manifest),
        "config": asdict(config),
        "clips_checked": len(checked),
        "clips_passed": len(passed),
        "clips_failed": len(failed),
        "anomalous_sources": len(anomalous_sources),
        "failure_reasons": dict(sorted(reason_counts.items())),
        "splits": {
            split: {
                "passed": split_pass[split],
                "failed": split_fail[split],
            }
            for split in SPLITS
        },
        "media_files_moved_or_deleted": 0,
    }
    atomic_write_text(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 2 if failed and args.fail_on_anomaly else 0


if __name__ == "__main__":
    raise SystemExit(main())
