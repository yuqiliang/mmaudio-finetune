#!/usr/bin/env python3
"""Apply source-level Excel decisions to a clip manifest without deleting media."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import Any
from xml.etree import ElementTree


SPLITS = ("train", "val", "test")
KEEP_DECISIONS = {"keep", "keep source", "include", "保留"}
EXCLUDE_DECISIONS = {
    "exclude",
    "exclude source",
    "exclude whole source",
    "remove",
    "排除",
    "删除",
}
PENDING_DECISIONS = {"", "not decided", "pending", "待定", "未决定"}
YES_VALUES = {"yes", "y", "true", "1", "是"}


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", newline="", dir=path.parent, delete=False
    ) as handle:
        handle.write(text)
        temp_path = Path(handle.name)
    os.replace(temp_path, path)


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


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def _column_index(cell_reference: str) -> int:
    letters = re.match(r"[A-Z]+", cell_reference.upper())
    if letters is None:
        raise ValueError(f"Invalid Excel cell reference: {cell_reference}")
    index = 0
    for letter in letters.group(0):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    try:
        root = ElementTree.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    return [
        "".join(node.text or "" for node in item.iter(f"{namespace}t"))
        for item in root.findall(f"{namespace}si")
    ]


def _sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    spreadsheet_ns = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    relationships_ns = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
    package_ns = "{http://schemas.openxmlformats.org/package/2006/relationships}"

    workbook = ElementTree.fromstring(archive.read("xl/workbook.xml"))
    relationship_id = None
    for sheet in workbook.findall(f".//{spreadsheet_ns}sheet"):
        if sheet.attrib.get("name") == sheet_name:
            relationship_id = sheet.attrib.get(f"{relationships_ns}id")
            break
    if relationship_id is None:
        available = [
            sheet.attrib.get("name", "")
            for sheet in workbook.findall(f".//{spreadsheet_ns}sheet")
        ]
        raise ValueError(f"Sheet {sheet_name!r} not found; available sheets: {available}")

    relationships = ElementTree.fromstring(
        archive.read("xl/_rels/workbook.xml.rels")
    )
    target = None
    for relationship in relationships.findall(f"{package_ns}Relationship"):
        if relationship.attrib.get("Id") == relationship_id:
            target = relationship.attrib.get("Target")
            break
    if target is None:
        raise ValueError(f"Missing relationship for sheet {sheet_name!r}")
    target_path = PurePosixPath(target.lstrip("/"))
    if not target_path.parts or target_path.parts[0] != "xl":
        target_path = PurePosixPath("xl") / target_path
    return str(target_path)


def _cell_value(cell: ElementTree.Element, shared: list[str]) -> str:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    cell_type = cell.attrib.get("t")
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.iter(f"{namespace}t"))
    value_node = cell.find(f"{namespace}v")
    if value_node is None or value_node.text is None:
        return ""
    value = value_node.text
    if cell_type == "s":
        return shared[int(value)]
    if cell_type == "b":
        return "TRUE" if value == "1" else "FALSE"
    return value


def read_xlsx_review(path: Path, sheet_name: str) -> list[dict[str, str]]:
    namespace = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
    with zipfile.ZipFile(path) as archive:
        shared = _shared_strings(archive)
        sheet = ElementTree.fromstring(archive.read(_sheet_path(archive, sheet_name)))

    matrix: list[dict[int, str]] = []
    for row in sheet.findall(f".//{namespace}sheetData/{namespace}row"):
        values: dict[int, str] = {}
        for cell in row.findall(f"{namespace}c"):
            reference = cell.attrib.get("r", "")
            values[_column_index(reference)] = _cell_value(cell, shared).strip()
        matrix.append(values)

    header_position = None
    headers: dict[int, str] = {}
    for position, row in enumerate(matrix):
        candidate = {index: value.strip() for index, value in row.items()}
        normalized = {value.lower() for value in candidate.values()}
        if "source file" in normalized and "decision" in normalized:
            header_position = position
            headers = candidate
            break
    if header_position is None:
        raise ValueError("Could not find an Excel header row containing Source File and Decision")

    records: list[dict[str, str]] = []
    for row in matrix[header_position + 1 :]:
        record = {header: row.get(index, "").strip() for index, header in headers.items()}
        if record.get("Source File", "").strip():
            records.append(record)
    return records


def read_review(path: Path, sheet_name: str) -> list[dict[str, str]]:
    suffix = path.suffix.lower()
    if suffix == ".xlsx":
        return read_xlsx_review(path, sheet_name)
    if suffix == ".csv":
        return read_csv(path)
    if suffix == ".tsv":
        return read_csv(path, delimiter="\t")
    raise ValueError("Review file must be .xlsx, .csv, or .tsv")


def normalized(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def classify_decision(row: dict[str, str]) -> str:
    decision = normalized(row.get("Decision"))
    music = normalized(row.get("Music"))
    if decision in KEEP_DECISIONS:
        if music in YES_VALUES:
            raise ValueError(
                f"{row.get('Source File')}: Music=Yes conflicts with Decision=Keep"
            )
        return "keep"
    if decision in EXCLUDE_DECISIONS:
        return "exclude"
    if decision in PENDING_DECISIONS:
        return "pending"
    raise ValueError(
        f"{row.get('Source File')}: unsupported Decision={row.get('Decision')!r}"
    )


def keep_from_seconds(row: dict[str, str], decision: str) -> float:
    raw_value = row.get("Keep From (s)", "").strip()
    if not raw_value:
        return 0.0
    if decision != "keep":
        raise ValueError(
            f"{row.get('Source File')}: Keep From (s) is only valid with Decision=Keep"
        )
    value = float(raw_value)
    if value < 0 or not math.isclose(value / 8.0, round(value / 8.0), abs_tol=1e-7):
        raise ValueError(
            f"{row.get('Source File')}: Keep From (s) must be a non-negative "
            "8-second boundary"
        )
    return value


def training_id(clip: dict[str, str]) -> str:
    source = (
        clip.get("combined_clip_id")
        or f"{clip.get('source_dataset_id', 'dataset')}__{clip['clip_id']}"
    )
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "__", source).strip("._-")
    if not cleaned:
        raise ValueError(f"Could not create a training id for {clip}")
    return cleaned


def source_key(row: dict[str, str]) -> str:
    return (
        row.get("source_key")
        or f"{row.get('source_dataset_id', '')}::{row['source_file']}"
    )


def union_fields(rows: list[dict[str, Any]], required: list[str]) -> list[str]:
    fields = list(required)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return fields


def write_split_outputs(output: Path, clips: list[dict[str, Any]], caption: str) -> None:
    clip_fields = union_fields(clips, ["training_id", "split", "absolute_path"])
    for split in SPLITS:
        split_rows = [row for row in clips if row["split"] == split]
        write_csv(
            output / f"final_{split}_clips_manifest.csv",
            split_rows,
            clip_fields,
        )
        atomic_write_text(
            output / f"final_{split}_paths.txt",
            "\n".join(str(row["absolute_path"]) for row in split_rows)
            + ("\n" if split_rows else ""),
        )
        tsv_rows = [
            {"id": row["training_id"], "caption": caption} for row in split_rows
        ]
        write_csv(
            output / f"video_{split}_ft.tsv",
            tsv_rows,
            ["id", "caption"],
            delimiter="\t",
        )


def apply_review(
    review_rows: list[dict[str, str]],
    source_rows: list[dict[str, str]],
    clip_rows: list[dict[str, str]],
    *,
    allow_pending: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    review_by_file: dict[str, dict[str, str]] = {}
    for row in review_rows:
        filename = row.get("Source File", "").strip()
        if filename in review_by_file:
            raise ValueError(f"Duplicate Excel review row: {filename}")
        review_by_file[filename] = row

    sources_by_file: dict[str, dict[str, str]] = {}
    for row in source_rows:
        filename = row["source_file"]
        if filename in sources_by_file:
            raise ValueError(f"Source filenames are not unique: {filename}")
        sources_by_file[filename] = row

    unknown = sorted(set(review_by_file) - set(sources_by_file))
    missing = sorted(set(sources_by_file) - set(review_by_file))
    if unknown:
        raise ValueError(f"Review contains unknown sources: {unknown[:5]}")
    if missing:
        raise ValueError(f"Review is missing sources: {missing[:5]}")

    kept: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    decision_by_key: dict[str, str] = {}
    keep_from_by_key: dict[str, float] = {}
    for filename, source in sources_by_file.items():
        review = review_by_file[filename]
        if review.get("Split", "").strip() and review["Split"].strip() != source["split"]:
            raise ValueError(
                f"{filename}: review split={review['Split']!r} does not match "
                f"manifest split={source['split']!r}"
            )
        decision = classify_decision(review)
        cutoff = keep_from_seconds(review, decision)
        result: dict[str, Any] = dict(source)
        result.update(
            {
                "review_decision": decision,
                "review_keep_from_seconds": cutoff if cutoff else "",
                "review_music": review.get("Music", "").strip(),
                "review_notes": review.get("Notes", "").strip(),
                "review_source_path": review.get("Source Path", "").strip(),
            }
        )
        decision_by_key[source_key(source)] = decision
        keep_from_by_key[source_key(source)] = cutoff
        if decision == "keep":
            kept.append(result)
        elif decision == "exclude":
            excluded.append(result)
        else:
            pending.append(result)

    if pending and not allow_pending:
        raise RuntimeError(
            f"{len(pending)} source decisions are pending. Complete the Excel Decision "
            "column or pass --allow-pending to build a safe interim manifest that omits them."
        )

    final_clips: list[dict[str, Any]] = []
    seen_training_ids: set[str] = set()
    for clip in clip_rows:
        key = source_key(clip)
        if key not in decision_by_key:
            raise ValueError(f"Clip references an unknown source: {key}")
        if decision_by_key[key] != "keep":
            continue
        if clip.get("status") and clip["status"] != "complete":
            raise ValueError(f"Kept clip is not complete: {clip.get('clip_id')}")
        cutoff = keep_from_by_key[key]
        if float(clip.get("source_start_seconds") or 0) < cutoff:
            continue
        result = dict(clip)
        result["training_id"] = training_id(clip)
        result["manual_review_status"] = "included"
        result["manual_keep_from_seconds"] = cutoff if cutoff else ""
        if result["training_id"] in seen_training_ids:
            raise ValueError(f"Duplicate training_id: {result['training_id']}")
        seen_training_ids.add(result["training_id"])
        final_clips.append(result)
    return kept, excluded, pending, final_clips


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--combined-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheet", default="Source Review")
    parser.add_argument("--caption", default="urban soundscape")
    parser.add_argument("--allow-pending", action="store_true")
    args = parser.parse_args()

    review_path = args.review.expanduser().resolve()
    combined_root = args.combined_root.expanduser().resolve()
    output = args.output.expanduser().resolve()
    source_rows = read_csv(combined_root / "metadata" / "source_split.csv")
    clip_rows = read_csv(combined_root / "metadata" / "clips_manifest.csv")
    review_rows = read_review(review_path, args.sheet)

    kept, excluded, pending, final_clips = apply_review(
        review_rows,
        source_rows,
        clip_rows,
        allow_pending=args.allow_pending,
    )
    output.mkdir(parents=True, exist_ok=True)
    source_fields = union_fields(
        kept + excluded + pending,
        ["source_key", "split", "source_file", "review_decision"],
    )
    clip_fields = union_fields(
        final_clips,
        ["training_id", "split", "source_file", "absolute_path"],
    )
    write_csv(output / "kept_sources.csv", kept, source_fields)
    write_csv(output / "excluded_sources.csv", excluded, source_fields)
    write_csv(output / "pending_sources.csv", pending, source_fields)
    write_csv(output / "final_clips_manifest.csv", final_clips, clip_fields)
    write_split_outputs(output, final_clips, args.caption)

    source_counts = Counter(row["split"] for row in kept)
    clip_counts = Counter(row["split"] for row in final_clips)
    clip_seconds: defaultdict[str, float] = defaultdict(float)
    for row in final_clips:
        clip_seconds[row["split"]] += float(
            row.get("actual_duration_seconds") or row.get("duration_seconds") or 0
        )
    cutoff_by_key = {
        source_key(row): float(row["review_keep_from_seconds"])
        for row in kept
        if row.get("review_keep_from_seconds") not in ("", None)
    }
    clips_omitted_before_cutoff = sum(
        source_key(row) in cutoff_by_key
        and float(row.get("source_start_seconds") or 0) < cutoff_by_key[source_key(row)]
        for row in clip_rows
    )
    summary = {
        "status": "interim" if pending else "final",
        "review_file": str(review_path),
        "source_videos": len(source_rows),
        "kept_sources": len(kept),
        "excluded_sources": len(excluded),
        "pending_sources": len(pending),
        "final_clips": len(final_clips),
        "clips_omitted_before_keep_from": clips_omitted_before_cutoff,
        "splits": {
            split: {
                "sources": source_counts[split],
                "clips": clip_counts[split],
                "clip_hours": round(clip_seconds[split] / 3600, 4),
            }
            for split in SPLITS
        },
        "media_files_moved_or_deleted": 0,
    }
    atomic_write_text(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
