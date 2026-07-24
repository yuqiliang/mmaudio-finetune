#!/usr/bin/env python3
"""Copy only manifest-listed clips to Drive with resume and verification."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any


SPLITS = ("train", "val", "test")
CHUNK_SIZE = 8 * 1024 * 1024


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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def destination_name(row: dict[str, str], source: Path) -> str:
    identifier = row.get("training_id") or row.get("combined_clip_id") or row.get("clip_id")
    if not identifier:
        raise ValueError(f"Manifest row has no training_id or clip_id: {row}")
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "__", identifier).strip("._-")
    if not cleaned:
        raise ValueError(f"Invalid training id: {identifier!r}")
    return cleaned + source.suffix.lower()


def file_matches(source: Path, destination: Path, verify: str) -> tuple[bool, str]:
    if not destination.is_file() or destination.stat().st_size != source.stat().st_size:
        return False, ""
    if verify == "size":
        return True, ""
    source_hash = sha256(source)
    return source_hash == sha256(destination), source_hash


def resume_copy(source: Path, destination: Path, verify: str) -> tuple[str, str]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    matches, source_hash = file_matches(source, destination, verify)
    if matches:
        return "skipped", source_hash
    if destination.exists():
        raise RuntimeError(
            f"Destination exists but does not match source: {destination}. "
            "Move it aside or pass --replace-mismatch."
        )

    partial = destination.with_suffix(destination.suffix + ".part")
    source_size = source.stat().st_size
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > source_size:
        partial.unlink()
        offset = 0
    with source.open("rb") as source_handle:
        source_handle.seek(offset)
        with partial.open("ab") as destination_handle:
            while chunk := source_handle.read(CHUNK_SIZE):
                destination_handle.write(chunk)
            destination_handle.flush()
            os.fsync(destination_handle.fileno())
    if partial.stat().st_size != source_size:
        raise RuntimeError(
            f"Incomplete copy for {source}: {partial.stat().st_size}/{source_size} bytes"
        )
    if verify == "sha256":
        source_hash = source_hash or sha256(source)
        if sha256(partial) != source_hash:
            partial.unlink()
            raise RuntimeError(f"SHA256 verification failed for {source}")
    os.replace(partial, destination)
    return ("resumed" if offset else "copied"), source_hash


def sync_row(
    row: dict[str, str],
    destination_root: Path,
    verify: str,
    retries: int,
    replace_mismatch: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source = Path(row["absolute_path"])
    if not source.is_file():
        raise FileNotFoundError(source)
    split = row.get("split", "")
    if split not in SPLITS:
        raise ValueError(f"Unexpected split={split!r} for {source}")
    destination = destination_root / split / destination_name(row, source)
    result: dict[str, Any] = dict(row)
    result["original_absolute_path"] = str(source)
    result["absolute_path"] = str(destination)
    result["relative_path"] = str(destination.relative_to(destination_root))
    if dry_run:
        result["sync_status"] = "dry_run"
        result["sync_sha256"] = ""
        return result

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if destination.exists() and replace_mismatch:
                matches, source_hash = file_matches(source, destination, verify)
                if matches:
                    status = "skipped"
                else:
                    replacement = destination.with_suffix(destination.suffix + ".part")
                    if replacement.exists():
                        replacement.unlink()
                    with source.open("rb") as source_handle, replacement.open("wb") as output:
                        while chunk := source_handle.read(CHUNK_SIZE):
                            output.write(chunk)
                        output.flush()
                        os.fsync(output.fileno())
                    if replacement.stat().st_size != source.stat().st_size:
                        raise RuntimeError(f"Replacement size mismatch for {destination}")
                    if verify == "sha256":
                        source_hash = source_hash or sha256(source)
                        if sha256(replacement) != source_hash:
                            replacement.unlink()
                            raise RuntimeError(f"Replacement SHA256 mismatch for {destination}")
                    os.replace(replacement, destination)
                    status = "replaced"
            else:
                status, source_hash = resume_copy(source, destination, verify)
            result["sync_status"] = status
            result["sync_sha256"] = source_hash
            result["byte_size"] = source.stat().st_size
            return result
        except Exception as error:
            last_error = error
            if attempt < retries:
                time.sleep(min(30, 2**attempt))
    assert last_error is not None
    raise last_error


def write_synced_outputs(
    metadata_dir: Path,
    synced: list[dict[str, Any]],
    caption: str,
) -> None:
    fields = union_fields(
        synced,
        ["training_id", "split", "absolute_path", "relative_path", "sync_status"],
    )
    write_csv(metadata_dir / "synced_clips_manifest.csv", synced, fields)
    for split in SPLITS:
        split_rows = [row for row in synced if row["split"] == split]
        write_csv(
            metadata_dir / f"synced_{split}_clips_manifest.csv",
            split_rows,
            fields,
        )
        atomic_write_text(
            metadata_dir / f"synced_{split}_paths.txt",
            "\n".join(str(row["absolute_path"]) for row in split_rows)
            + ("\n" if split_rows else ""),
        )
        write_csv(
            metadata_dir / f"video_{split}_ft.tsv",
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
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--verify", choices=("size", "sha256"), default="sha256")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--caption", default="urban soundscape")
    parser.add_argument("--replace-mismatch", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest = args.manifest.expanduser().resolve()
    destination = args.destination.expanduser().resolve()
    rows = read_csv(manifest)
    if not rows:
        raise ValueError(f"Manifest is empty: {manifest}")
    destination.mkdir(parents=True, exist_ok=True)
    planned_paths = [
        destination
        / row.get("split", "")
        / destination_name(row, Path(row["absolute_path"]))
        for row in rows
    ]
    if len(planned_paths) != len(set(planned_paths)):
        duplicates = [
            str(path)
            for path, count in Counter(planned_paths).items()
            if count > 1
        ]
        raise ValueError(f"Duplicate destination paths: {duplicates[:5]}")

    def run(item: tuple[int, dict[str, str]]) -> tuple[int, dict[str, Any], str]:
        index, row = item
        try:
            synced = sync_row(
                row,
                destination,
                args.verify,
                args.retries,
                args.replace_mismatch,
                args.dry_run,
            )
            print(
                f"[{index + 1}/{len(rows)}] {synced['sync_status']}: "
                f"{synced.get('training_id', synced.get('clip_id'))}"
            )
            return index, synced, ""
        except Exception as error:
            print(
                f"[{index + 1}/{len(rows)}] failed: "
                f"{row.get('training_id', row.get('clip_id'))}: {error}"
            )
            return index, dict(row), str(error)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        results = list(executor.map(run, enumerate(rows)))
    results.sort(key=lambda item: item[0])
    synced = [row for _, row, error in results if not error]
    failures = [
        {
            "training_id": row.get("training_id", row.get("clip_id", "")),
            "absolute_path": row.get("absolute_path", ""),
            "error": error,
        }
        for _, row, error in results
        if error
    ]

    metadata_dir = destination / "metadata"
    write_synced_outputs(metadata_dir, synced, args.caption)
    write_csv(
        metadata_dir / "sync_failures.csv",
        failures,
        ["training_id", "absolute_path", "error"],
    )
    status_counts = Counter(row["sync_status"] for row in synced)
    split_counts = Counter(row["split"] for row in synced)
    split_bytes: defaultdict[str, int] = defaultdict(int)
    for row in synced:
        split_bytes[row["split"]] += int(row.get("byte_size") or 0)
    summary = {
        "source_manifest": str(manifest),
        "destination": str(destination),
        "verification": args.verify,
        "dry_run": args.dry_run,
        "requested_files": len(rows),
        "synced_files": len(synced),
        "failed_files": len(failures),
        "status_counts": dict(status_counts),
        "splits": {
            split: {
                "files": split_counts[split],
                "bytes": split_bytes[split],
            }
            for split in SPLITS
        },
        "source_files_moved_or_deleted": 0,
    }
    atomic_write_text(
        metadata_dir / "sync_summary.json",
        json.dumps(summary, indent=2) + "\n",
    )
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
