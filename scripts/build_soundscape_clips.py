#!/usr/bin/env python3
"""Create non-overlapping 8-second clips without modifying source videos."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_SOURCE = Path("/Volumes/SSID IVR Study 1/Yuqi/Fusion Export")
DEFAULT_OUTPUT = Path("/Volumes/SSID IVR Study 1/Yuqi/Fusion Export Clips 8s")
DEFAULT_DATASET_ID = "soundscape_nonoverlap_8s_v1"
CLIP_SECONDS = 8
TARGET_WIDTH = 1280
TARGET_HEIGHT = 720
TARGET_FPS = 25
TARGET_SAMPLE_RATE = 44100
TARGET_AUDIO_CHANNELS = 1
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".wmv", ".m4v"}
SPLIT_FRACTIONS = {"train": 0.8, "val": 0.1, "test": 0.1}
SPLIT_ORDER = ("train", "val", "test")


@dataclass(frozen=True)
class SourceRecord:
    split: str
    source_file: str
    duration_seconds: float
    expected_clips: int
    width: int
    height: int
    frame_rate: str


@dataclass(frozen=True)
class ClipRecord:
    dataset_id: str
    clip_id: str
    split: str
    source_file: str
    source_start_seconds: int
    duration_seconds: int
    relative_path: str


def run_json(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return json.loads(result.stdout)


def probe_source(path: Path) -> dict[str, Any]:
    payload = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=index,codec_type,codec_name,width,height,r_frame_rate,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise RuntimeError(f"Missing video or audio stream: {path.name}")
    if int(audio.get("channels", 0)) < 1:
        raise RuntimeError(f"Invalid audio stream: {path.name}")
    return {
        "duration": float(payload["format"]["duration"]),
        "width": int(video["width"]),
        "height": int(video["height"]),
        "frame_rate": str(video["r_frame_rate"]),
    }


def probe_clip(path: Path) -> dict[str, Any]:
    payload = run_json(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type,codec_name,width,height,r_frame_rate,pix_fmt,sample_rate,channels",
            "-of",
            "json",
            str(path),
        ]
    )
    streams = payload.get("streams", [])
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), None)
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    if video is None or audio is None:
        raise RuntimeError(f"Missing output video or audio stream: {path.name}")
    return {
        "duration": float(payload["format"]["duration"]),
        "video_codec": video.get("codec_name"),
        "width": int(video.get("width", 0)),
        "height": int(video.get("height", 0)),
        "frame_rate": video.get("r_frame_rate"),
        "pixel_format": video.get("pix_fmt"),
        "audio_codec": audio.get("codec_name"),
        "sample_rate": int(audio.get("sample_rate", 0)),
        "channels": int(audio.get("channels", 0)),
    }


def split_score(totals: dict[str, float], targets: dict[str, float]) -> float:
    return sum(((totals[name] - targets[name]) / targets[name]) ** 2 for name in SPLIT_ORDER)


def assign_splits(probes: list[tuple[Path, dict[str, Any]]]) -> dict[str, str]:
    source_count = len(probes)
    eval_count = max(1, round(source_count * SPLIT_FRACTIONS["val"]))
    caps = {
        "train": source_count - 2 * eval_count,
        "val": eval_count,
        "test": eval_count,
    }
    total_duration = sum(item[1]["duration"] for item in probes)
    targets = {name: total_duration * SPLIT_FRACTIONS[name] for name in SPLIT_ORDER}
    totals = {name: 0.0 for name in SPLIT_ORDER}
    members: dict[str, list[tuple[Path, dict[str, Any]]]] = {name: [] for name in SPLIT_ORDER}

    for item in sorted(probes, key=lambda value: (-value[1]["duration"], value[0].name)):
        available = [name for name in SPLIT_ORDER if len(members[name]) < caps[name]]
        chosen = min(available, key=lambda name: (totals[name] / targets[name], SPLIT_ORDER.index(name)))
        members[chosen].append(item)
        totals[chosen] += item[1]["duration"]

    # Pairwise swaps preserve source counts while tightening duration balance.
    while True:
        current_score = split_score(totals, targets)
        best: tuple[float, str, int, str, int] | None = None
        for left_pos, left_name in enumerate(SPLIT_ORDER):
            for right_name in SPLIT_ORDER[left_pos + 1 :]:
                for left_index, left_item in enumerate(members[left_name]):
                    for right_index, right_item in enumerate(members[right_name]):
                        candidate = dict(totals)
                        candidate[left_name] += right_item[1]["duration"] - left_item[1]["duration"]
                        candidate[right_name] += left_item[1]["duration"] - right_item[1]["duration"]
                        score = split_score(candidate, targets)
                        if score + 1e-12 < current_score and (best is None or score < best[0]):
                            best = (score, left_name, left_index, right_name, right_index)
        if best is None:
            break
        _, left_name, left_index, right_name, right_index = best
        left_item = members[left_name][left_index]
        right_item = members[right_name][right_index]
        members[left_name][left_index], members[right_name][right_index] = right_item, left_item
        totals[left_name] += right_item[1]["duration"] - left_item[1]["duration"]
        totals[right_name] += left_item[1]["duration"] - right_item[1]["duration"]

    return {item[0].name: name for name in SPLIT_ORDER for item in members[name]}


def atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def build_plan(
    source_dir: Path,
    output_dir: Path,
    dataset_id: str,
) -> tuple[list[SourceRecord], list[ClipRecord]]:
    source_paths = sorted(
        path
        for path in source_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not source_paths:
        raise RuntimeError(f"No supported video files found in {source_dir}")
    stems = [path.stem for path in source_paths]
    if len(stems) != len(set(stems)):
        raise RuntimeError("Source file stems must be unique")

    print(f"Probing {len(source_paths)} source videos...", flush=True)
    probes = [(path, probe_source(path)) for path in source_paths]
    assignments = assign_splits(probes)

    sources: list[SourceRecord] = []
    clips: list[ClipRecord] = []
    for path, probe in probes:
        expected = math.floor((probe["duration"] + 1e-6) / CLIP_SECONDS)
        split = assignments[path.name]
        sources.append(
            SourceRecord(
                split=split,
                source_file=path.name,
                duration_seconds=round(probe["duration"], 6),
                expected_clips=expected,
                width=probe["width"],
                height=probe["height"],
                frame_rate=probe["frame_rate"],
            )
        )
        for index in range(expected):
            clip_id = f"{path.stem}__{index:06d}"
            clips.append(
                ClipRecord(
                    dataset_id=dataset_id,
                    clip_id=clip_id,
                    split=split,
                    source_file=path.name,
                    source_start_seconds=index * CLIP_SECONDS,
                    duration_seconds=CLIP_SECONDS,
                    relative_path=f"{split}/{clip_id}.mp4",
                )
            )

    sources.sort(key=lambda item: (SPLIT_ORDER.index(item.split), item.source_file))
    clips.sort(key=lambda item: (SPLIT_ORDER.index(item.split), item.source_file, item.source_start_seconds))
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        metadata_dir / "source_split.csv",
        [asdict(item) for item in sources],
        list(asdict(sources[0]).keys()),
    )
    write_csv(
        metadata_dir / "planned_clips.csv",
        [asdict(item) for item in clips],
        list(asdict(clips[0]).keys()),
    )
    return sources, clips


def encoder_available(name: str) -> bool:
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return name in result.stdout


def choose_encoder(requested: str) -> str:
    if requested != "auto":
        return requested
    if platform.system() == "Darwin" and encoder_available("h264_videotoolbox"):
        return "h264_videotoolbox"
    return "libx264"


def log(message: str, log_path: Path) -> None:
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(timestamped + "\n")


def config_fingerprint(dataset_id: str, encoder: str, quality: int, crf: int) -> str:
    config = {
        "dataset_id": dataset_id,
        "clip_seconds": CLIP_SECONDS,
        "target_width": TARGET_WIDTH,
        "target_height": TARGET_HEIGHT,
        "target_fps": TARGET_FPS,
        "encoder": encoder,
        "quality": quality,
        "crf": crf,
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "audio_sample_rate": TARGET_SAMPLE_RATE,
        "audio_channels": TARGET_AUDIO_CHANNELS,
    }
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest()


def marker_is_valid(marker_path: Path, source_path: Path, fingerprint: str, expected: int) -> bool:
    if not marker_path.exists():
        return False
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        source_stat = source_path.stat()
        if marker["config_fingerprint"] != fingerprint:
            return False
        if marker["source_size"] != source_stat.st_size:
            return False
        if marker["source_mtime_ns"] != source_stat.st_mtime_ns:
            return False
        if len(marker["clips"]) != expected:
            return False
        return all(Path(item["absolute_path"]).is_file() and Path(item["absolute_path"]).stat().st_size > 0 for item in marker["clips"])
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def encode_source(
    source: SourceRecord,
    source_dir: Path,
    output_dir: Path,
    dataset_id: str,
    encoder: str,
    quality: int,
    crf: int,
    fingerprint: str,
    log_path: Path,
) -> None:
    source_path = source_dir / source.source_file
    source_stat_before = source_path.stat()
    destination = output_dir / source.split
    destination.mkdir(parents=True, exist_ok=True)
    state_dir = output_dir / "metadata" / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    marker_path = state_dir / f"{source_path.stem}.json"
    if marker_is_valid(marker_path, source_path, fingerprint, source.expected_clips):
        log(f"SKIP {source.source_file}: {source.expected_clips} clips already complete", log_path)
        return

    for old_clip in destination.glob(f"{source_path.stem}__*.mp4"):
        old_clip.unlink()

    usable_seconds = source.expected_clips * CLIP_SECONDS
    pattern = destination / f"{source_path.stem}__%06d.mp4"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-y",
        "-i",
        str(source_path),
        "-t",
        str(usable_seconds),
        "-map",
        "0:v:0",
        "-map",
        "0:a:0",
        "-map_metadata",
        "-1",
        "-c:v",
        encoder,
        "-vf",
        f"scale={TARGET_WIDTH}:{TARGET_HEIGHT},fps={TARGET_FPS},setsar=1",
    ]
    if encoder == "h264_videotoolbox":
        command += [
            "-q:v",
            str(quality),
            "-profile:v",
            "high",
            "-allow_sw",
            "1",
            "-pix_fmt",
            "yuv420p",
        ]
    else:
        command += ["-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p"]
    command += [
        "-force_key_frames",
        f"expr:gte(t,n_forced*{CLIP_SECONDS})",
        "-fps_mode",
        "cfr",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-ac",
        str(TARGET_AUDIO_CHANNELS),
        "-f",
        "segment",
        "-segment_time",
        str(CLIP_SECONDS),
        "-segment_time_delta",
        "0.02",
        "-reset_timestamps",
        "1",
        "-segment_start_number",
        "0",
        "-segment_format",
        "mp4",
        "-segment_format_options",
        "movflags=+faststart",
        "-progress",
        "pipe:1",
        "-nostats",
        str(pattern),
    ]

    log(
        f"START {source.source_file}: {source.expected_clips} clips, {usable_seconds}s, {encoder}",
        log_path,
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    last_reported = -60
    captured_errors: list[str] = []
    try:
        assert process.stdout is not None
        for raw_line in process.stdout:
            line = raw_line.strip()
            if line.startswith("out_time="):
                time_text = line.split("=", 1)[1]
                try:
                    hours, minutes, seconds = time_text.split(":")
                    encoded_seconds = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                except ValueError:
                    continue
                if encoded_seconds - last_reported >= 60:
                    log(
                        f"PROGRESS {source.source_file}: {min(encoded_seconds, usable_seconds):.0f}/{usable_seconds}s",
                        log_path,
                    )
                    last_reported = int(encoded_seconds)
            elif line and "=" not in line:
                captured_errors.append(line)
        return_code = process.wait()
    except KeyboardInterrupt:
        process.terminate()
        process.wait()
        raise
    if return_code != 0:
        detail = " | ".join(captured_errors[-5:])
        raise RuntimeError(f"FFmpeg failed for {source.source_file}: {detail}")

    expected_paths = [destination / f"{source_path.stem}__{index:06d}.mp4" for index in range(source.expected_clips)]
    extra_paths = sorted(destination.glob(f"{source_path.stem}__*.mp4"))[source.expected_clips :]
    for extra_path in extra_paths:
        extra_path.unlink()
    missing = [path.name for path in expected_paths if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing outputs for {source.source_file}: {missing[:5]}")

    clip_states = []
    for clip_path in expected_paths:
        clip_info = probe_clip(clip_path)
        actual_duration = clip_info["duration"]
        if not 7.70 <= actual_duration <= 8.30:
            raise RuntimeError(f"Unexpected duration {actual_duration:.3f}s: {clip_path.name}")
        expected_video = {
            "video_codec": "h264",
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "frame_rate": f"{TARGET_FPS}/1",
            "pixel_format": "yuv420p",
        }
        expected_audio = {
            "audio_codec": "aac",
            "sample_rate": TARGET_SAMPLE_RATE,
            "channels": TARGET_AUDIO_CHANNELS,
        }
        for key, expected_value in {**expected_video, **expected_audio}.items():
            if clip_info[key] != expected_value:
                raise RuntimeError(
                    f"Unexpected {key}={clip_info[key]!r}, expected {expected_value!r}: {clip_path.name}"
                )
        clip_states.append(
            {
                "absolute_path": str(clip_path),
                "relative_path": str(clip_path.relative_to(output_dir)),
                "actual_duration_seconds": round(actual_duration, 6),
                "byte_size": clip_path.stat().st_size,
                "width": clip_info["width"],
                "height": clip_info["height"],
                "frame_rate": clip_info["frame_rate"],
                "sample_rate": clip_info["sample_rate"],
                "channels": clip_info["channels"],
            }
        )

    source_stat_after = source_path.stat()
    if (source_stat_before.st_size, source_stat_before.st_mtime_ns) != (
        source_stat_after.st_size,
        source_stat_after.st_mtime_ns,
    ):
        raise RuntimeError(f"Source changed during processing: {source.source_file}")
    marker = {
        "dataset_id": dataset_id,
        "config_fingerprint": fingerprint,
        "source_file": source.source_file,
        "source_size": source_stat_after.st_size,
        "source_mtime_ns": source_stat_after.st_mtime_ns,
        "clips": clip_states,
    }
    atomic_write_text(marker_path, json.dumps(marker, indent=2, sort_keys=True) + "\n")
    log(f"DONE {source.source_file}: {len(clip_states)} clips validated", log_path)


def write_final_manifest(output_dir: Path, planned_clips: list[ClipRecord]) -> tuple[int, int]:
    state_dir = output_dir / "metadata" / "state"
    state_by_path: dict[str, dict[str, Any]] = {}
    if state_dir.exists():
        for marker_path in state_dir.glob("*.json"):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            for clip in marker.get("clips", []):
                state_by_path[clip["relative_path"]] = clip
    rows = []
    complete = 0
    for clip in planned_clips:
        state = state_by_path.get(clip.relative_path)
        status = "complete" if state else "pending"
        complete += int(state is not None)
        row = asdict(clip)
        row.update(
            {
                "actual_duration_seconds": state["actual_duration_seconds"] if state else "",
                "byte_size": state["byte_size"] if state else "",
                "status": status,
            }
        )
        rows.append(row)
    write_csv(output_dir / "metadata" / "clips_manifest.csv", rows, list(rows[0].keys()))
    return complete, len(planned_clips)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dataset-id", default=DEFAULT_DATASET_ID)
    parser.add_argument(
        "--encoder",
        choices=("auto", "h264_videotoolbox", "libx264"),
        default="auto",
    )
    parser.add_argument("--quality", type=int, default=65, help="VideoToolbox quality (1-100)")
    parser.add_argument("--crf", type=int, default=20, help="libx264 CRF")
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_dir = args.source.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise RuntimeError("Output must not be the source directory or a child of it")
    if not source_dir.is_dir():
        raise RuntimeError(f"Source directory does not exist: {source_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLIT_ORDER:
        (output_dir / split).mkdir(exist_ok=True)

    sources, clips = build_plan(source_dir, output_dir, args.dataset_id)
    encoder = choose_encoder(args.encoder)
    fingerprint = config_fingerprint(args.dataset_id, encoder, args.quality, args.crf)
    config = {
        "dataset_id": args.dataset_id,
        "source_directory": str(source_dir),
        "output_directory": str(output_dir),
        "source_is_read_only": True,
        "clip_seconds": CLIP_SECONDS,
        "stride_seconds": CLIP_SECONDS,
        "overlap_seconds": 0,
        "drop_incomplete_tail": True,
        "split_strategy": "source-level duration-balanced 80/10/10",
        "normalization": {
            "width": TARGET_WIDTH,
            "height": TARGET_HEIGHT,
            "fps": TARGET_FPS,
            "pixel_format": "yuv420p",
            "audio_sample_rate": TARGET_SAMPLE_RATE,
            "audio_channels": TARGET_AUDIO_CHANNELS,
        },
        "encoder": encoder,
        "videotoolbox_quality": args.quality,
        "libx264_crf": args.crf,
        "expected_sources": len(sources),
        "expected_clips": len(clips),
        "config_fingerprint": fingerprint,
    }
    atomic_write_text(output_dir / "metadata" / "dataset_config.json", json.dumps(config, indent=2) + "\n")

    summary: dict[str, dict[str, float | int]] = {}
    for split in SPLIT_ORDER:
        selected = [source for source in sources if source.split == split]
        summary[split] = {
            "sources": len(selected),
            "source_hours": round(sum(source.duration_seconds for source in selected) / 3600, 4),
            "clips": sum(source.expected_clips for source in selected),
        }
    print(json.dumps({"plan": summary, "total_clips": len(clips)}, indent=2), flush=True)
    if args.plan_only:
        print("Plan created; no clips encoded.", flush=True)
        return 0

    log_path = output_dir / "metadata" / "processing.log"
    failures: list[dict[str, str]] = []
    for index, source in enumerate(sources, start=1):
        log(f"SOURCE {index}/{len(sources)} ({source.split})", log_path)
        try:
            encode_source(
                source,
                source_dir,
                output_dir,
                args.dataset_id,
                encoder,
                args.quality,
                args.crf,
                fingerprint,
                log_path,
            )
        except KeyboardInterrupt:
            complete, total = write_final_manifest(output_dir, clips)
            log(f"INTERRUPTED: {complete}/{total} clips complete", log_path)
            return 130
        except Exception as error:  # Continue so one damaged source does not discard the rest of the run.
            failures.append({"source_file": source.source_file, "error": str(error)})
            log(f"FAILED {source.source_file}: {error}", log_path)
        complete, total = write_final_manifest(output_dir, clips)
        log(f"DATASET PROGRESS: {complete}/{total} clips complete", log_path)

    write_csv(
        output_dir / "metadata" / "failures.csv",
        failures,
        ["source_file", "error"],
    )
    complete, total = write_final_manifest(output_dir, clips)
    log(f"FINISHED: {complete}/{total} clips complete, {len(failures)} failed sources", log_path)
    return 0 if complete == total and not failures else 1


if __name__ == "__main__":
    sys.exit(main())
