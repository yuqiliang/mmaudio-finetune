#!/usr/bin/env python3
"""Offline, review-gated B0 acoustic caption registry; never access media or models.

init creates an all-pending JSON registry bound to one exact CSV manifest.
Humans retain caption_candidate and edit caption_final plus explicit review fields
in a NEW registry revision. validate permits pending annotations by default;
export requires approved acoustic-only captions for every input manifest row.
Export readiness is not tokenizer, official encoder, or training readiness.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import unicodedata


SCHEMA_VERSION = 1
SAFE_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
SPLITS = {"train", "val", "test"}
CAPTION_SOURCES = {"missing", "manifest", "human", "soundscaper", "model"}
IDENTITY_FIELDS = (
    "training_id", "clip_id", "source_recording_id", "source_key", "source_dataset_id",
    "source_file", "site_id", "split", "source_start_seconds", "actual_duration_seconds",
    "duration_seconds", "qc_duration_seconds", "absolute_path", "original_absolute_path",
    "relative_path", "path", "sha256", "sync_sha256",
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")


def version(value: str) -> str:
    if not isinstance(value, str) or not SAFE_VERSION.fullmatch(value):
        raise ValueError("Explicit safe non-empty version required (letters, digits, . _ -)")
    return value


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: expected finite number, not boolean")
    try:
        number = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name}: expected finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{name}: expected finite number")
    return number


def clean_text(value: object, name: str, *, required: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name}: expected string")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value):
        raise ValueError(f"{name}: control/format characters are not allowed")
    if required and not value.strip():
        raise ValueError(f"{name}: non-empty value required")
    return value


def canonical_id(row: dict) -> str:
    identifier = (row.get("training_id") or row.get("clip_id") or "").strip()
    if not SAFE_VERSION.fullmatch(identifier):
        raise ValueError(f"Missing or unsafe canonical ID (training_id else clip_id): {identifier!r}")
    return identifier


def source_id(row: dict) -> str:
    result = (row.get("source_recording_id") or row.get("source_key") or "").strip()
    if result:
        return result
    if not row.get("source_file", "").strip():
        raise ValueError("source_recording_id/source_key/source_file required")
    return f"{row.get('source_dataset_id', '')}::{row['source_file']}"


def identity(row: dict) -> dict:
    start = finite(row.get("source_start_seconds"), "source_start_seconds")
    # Annotation windows follow the declared clip interval, not AAC/container
    # padding (an 8-second clip can have an actual duration such as 8.011).
    duration_field = next((key for key in (
        "duration_seconds", "actual_duration_seconds", "qc_duration_seconds") if row.get(key)), "")
    duration = finite(row.get(duration_field), "duration_seconds")
    if start < 0 or duration <= 0:
        raise ValueError("Clip start must be non-negative and duration must be positive")
    fingerprint = row.get("sync_sha256") or row.get("sha256") or ""
    if fingerprint and not re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint):
        raise ValueError("Recorded media fingerprint must be a SHA256 hex digest")
    return {"canonical_id": canonical_id(row), "source_recording_id": source_id(row),
            "site_id": row.get("site_id", ""), "split": row.get("split", ""),
            "start_seconds": start, "duration_seconds": duration, "duration_source_field": duration_field,
            "media_fingerprint": {"algorithm": "sha256" if fingerprint else "",
                                  "value": fingerprint, "verified_by_this_tool": False},
            "manifest_fields": {key: row[key] for key in IDENTITY_FIELDS if key in row}}


def read_manifest(path: Path) -> tuple[bytes, list[str], list[dict]]:
    data = path.read_bytes()
    reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig"), newline=""))
    fields = reader.fieldnames or []
    if not fields or len(set(fields)) != len(fields) or any(not f.strip() for f in fields):
        raise ValueError("Missing or duplicate CSV column names")
    rows = list(reader)
    if not rows:
        raise ValueError("Empty input manifest")
    ids: set[str] = set()
    sources: dict[str, str] = {}
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Malformed CSV row: wrong column count")
        identifier = canonical_id(row)
        if identifier in ids:
            raise ValueError(f"Duplicate canonical ID: {identifier}")
        ids.add(identifier)
        if row.get("split") not in SPLITS:
            raise ValueError(f"{identifier}: split must be train, val, or test")
        if row.get("manual_review_status") != "included" or row.get("qc_status") != "pass":
            raise ValueError(f"{identifier}: final human-reviewed, QC-passed manifest required")
        if row.get("clapboard_final_status", "included") != "included":
            raise ValueError(f"{identifier}: clapboard review is not final/included")
        source = source_id(row)
        if source in sources and sources[source] != row["split"]:
            raise ValueError(f"Source split leakage: {source}")
        sources[source] = row["split"]
        identity(row)
    return data, fields, rows


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_registry(path: Path) -> tuple[bytes, dict]:
    data = path.read_bytes()
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid non-finite JSON number: {value}")
    value = json.loads(data, object_pairs_hook=_unique_object, parse_constant=invalid_constant)
    if not isinstance(value, dict):
        raise ValueError("Registry must be a JSON object")
    return data, value


def init_registry(manifest: Path, registry_version: str, output: Path,
                  *, dry_run: bool = False) -> dict:
    version(registry_version)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing to overwrite registry: {output}")
    data, _, rows = read_manifest(manifest)
    records = []
    for row in rows:
        candidate = row.get("caption") or row.get("label") or ""
        clean_text(candidate, "original caption candidate")
        records.append({
            "canonical_id": canonical_id(row), "identity": identity(row),
            "manifest_row_sha256": digest(canonical_json(row)),
            "original_caption": {"label": row.get("label", ""), "caption": row.get("caption", "")},
            "caption_candidate": candidate, "caption_final": "",
            "caption_review_status": "pending", "caption_source": "manifest" if candidate else "missing",
            "generation": {"model": "", "prompt_version": "", "code_commit": "",
                           "weights_sha256": "", "run_id": ""},
            "reviewer": "", "reviewed_at": "", "acoustic_only_reviewed": False,
            "perception": {"label_source": "missing", "temporal_scope": None, "scale": None,
                           "dimension_ids": [], "scores": {}, "annotation_version": "",
                           "annotator_or_model": "", "inherited_from": ""},
        })
    registry = {"schema_version": SCHEMA_VERSION, "registry_version": registry_version,
                "created_at": utc_now(), "condition_arm": "B0_acoustic_only",
                "manifest": {"path": str(manifest.resolve()), "sha256": digest(data), "row_count": len(rows)},
                "records": records}
    validate_registry(registry, data, rows, registry_version)
    summary = {"status": "PREVIEW" if dry_run else "WAITING_FOR_CAPTION_REVIEW", "rows": len(rows),
               "registry_version": registry_version, "output": str(output.resolve()),
               "dry_run": dry_run, "media_files_read_or_changed": 0}
    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            json.dump(registry, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return summary


def validate_perception(value: object, record_identity: dict) -> None:
    if not isinstance(value, dict):
        raise ValueError("perception metadata must be an object (or missing-label object)")
    source = value.get("label_source")
    if source not in {"missing", "human", "model", "inherited"}:
        raise ValueError("Invalid perception label_source")
    if source == "missing":
        if any(value.get(k) for k in ("scores", "dimension_ids", "temporal_scope", "scale",
                                      "annotation_version", "annotator_or_model", "inherited_from")):
            raise ValueError("Missing perception labels must not contain fabricated/partial metadata")
        return
    for field in ("annotation_version", "annotator_or_model"):
        clean_text(value.get(field), f"perception.{field}", required=True)
    dimensions, scores, scale = value.get("dimension_ids"), value.get("scores"), value.get("scale")
    if (not isinstance(dimensions, list) or not dimensions or
            any(not isinstance(d, str) or not d.strip() for d in dimensions) or
            len(set(dimensions)) != len(dimensions) or not isinstance(scores, dict) or
            set(scores) != set(dimensions)):
        raise ValueError("perception dimension_ids and scores must match exactly, without duplicates")
    if not isinstance(scale, dict):
        raise ValueError("perception scale metadata required")
    clean_text(scale.get("name"), "perception.scale.name", required=True)
    lo, hi = finite(scale.get("min"), "scale.min"), finite(scale.get("max"), "scale.max")
    if lo >= hi:
        raise ValueError("perception scale.min must be less than scale.max")
    for dimension, score in scores.items():
        clean_text(dimension, "perception dimension")
        if not lo <= finite(score, f"score.{dimension}") <= hi:
            raise ValueError(f"Perception score outside declared scale: {dimension}")
    scope = value.get("temporal_scope")
    if not isinstance(scope, dict) or scope.get("unit") not in {"clip", "segment", "source_recording", "site"}:
        raise ValueError("perception temporal_scope with explicit unit required")
    clean_text(scope.get("scope_id"), "perception.temporal_scope.scope_id", required=True)
    if scope["unit"] == "site":
        if not record_identity["site_id"] or scope["scope_id"] != record_identity["site_id"]:
            raise ValueError("Perception site scope does not match clip site_id")
    else:
        start = finite(scope.get("start_seconds"), "perception scope start")
        end = finite(scope.get("end_seconds"), "perception scope end")
        if start < 0 or end <= start:
            raise ValueError("Invalid perception temporal range")
        clip_start = record_identity["start_seconds"]
        clip_end = clip_start + record_identity["duration_seconds"]
        if scope["unit"] == "clip":
            if (scope["scope_id"] != record_identity["canonical_id"] or
                    abs(start - clip_start) > 1e-6 or abs(end - clip_end) > 1e-6):
                raise ValueError("Perception clip temporal scope does not match the clip")
        elif scope["scope_id"] != record_identity["source_recording_id"] or start > clip_start or end < clip_end:
            raise ValueError("Inherited perception range/source must cover the clip")
    if scope["unit"] != "clip" and source != "inherited":
        raise ValueError("Source/segment/site labels assigned to clips must be marked inherited")
    if source == "inherited":
        clean_text(value.get("inherited_from"), "perception.inherited_from", required=True)
    elif value.get("inherited_from"):
        raise ValueError("inherited_from is only valid for inherited perception labels")


def validate_registry(registry: dict, manifest_data: bytes, rows: list[dict],
                      registry_version: str, *, require_approved: bool = False) -> dict:
    version(registry_version)
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported registry schema_version")
    if registry.get("registry_version") != registry_version:
        raise ValueError("Registry version mismatch; use the explicit intended revision")
    if registry.get("condition_arm") != "B0_acoustic_only":
        raise ValueError("Only B0_acoustic_only export is implemented")
    binding = registry.get("manifest")
    if (not isinstance(binding, dict) or binding.get("sha256") != digest(manifest_data)
            or binding.get("row_count") != len(rows)):
        raise ValueError("Exact input manifest SHA256/row_count mismatch; initialise a new registry")
    records = registry.get("records")
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        raise ValueError("records must be a list of objects")
    by_id = {}
    for record in records:
        identifier = record.get("canonical_id")
        if not isinstance(identifier, str) or not identifier:
            raise ValueError("Missing record canonical_id")
        if identifier in by_id:
            raise ValueError(f"Duplicate registry ID: {identifier}")
        by_id[identifier] = record
    expected = {canonical_id(row) for row in rows}
    if set(by_id) != expected:
        raise ValueError(f"Registry exact coverage mismatch: missing={sorted(expected - set(by_id))}, "
                         f"unknown={sorted(set(by_id) - expected)}")
    counts = {"pending": 0, "approved": 0, "rejected": 0}
    for row in rows:
        identifier = canonical_id(row)
        record = by_id[identifier]
        actual_identity = identity(row)
        if record.get("identity") != actual_identity or record.get("manifest_row_sha256") != digest(canonical_json(row)):
            raise ValueError(f"{identifier}: source/time/split/path/identity mismatch")
        if record.get("original_caption") != {"label": row.get("label", ""), "caption": row.get("caption", "")}:
            raise ValueError(f"{identifier}: original caption provenance mismatch")
        candidate = clean_text(record.get("caption_candidate"), f"{identifier}.caption_candidate")
        final = clean_text(record.get("caption_final"), f"{identifier}.caption_final")
        status = record.get("caption_review_status")
        if status not in counts:
            raise ValueError(f"{identifier}: invalid caption_review_status")
        counts[status] += 1
        source = record.get("caption_source")
        if source not in CAPTION_SOURCES or (source == "missing" and (candidate or final)):
            raise ValueError(f"{identifier}: missing/invalid caption provenance")
        generation = record.get("generation")
        if not isinstance(generation, dict):
            raise ValueError(f"{identifier}: generation provenance object required")
        for key in ("model", "prompt_version"):
            clean_text(generation.get(key), f"{identifier}.generation.{key}", required=source in {"soundscaper", "model"})
        if source in {"soundscaper", "model"}:
            clean_text(candidate, f"{identifier}.caption_candidate (retained model proposal)", required=True)
        for key in ("code_commit", "weights_sha256", "run_id"):
            clean_text(generation.get(key, ""), f"{identifier}.generation.{key}")
        if generation.get("weights_sha256") and not re.fullmatch(r"[0-9a-fA-F]{64}", generation["weights_sha256"]):
            raise ValueError(f"{identifier}: generation.weights_sha256 must be a SHA256 digest")
        if source == "manifest" and candidate != (row.get("caption") or row.get("label") or ""):
            raise ValueError(f"{identifier}: imported manifest candidate changed; record new caption_source")
        if not isinstance(record.get("acoustic_only_reviewed"), bool):
            raise ValueError(f"{identifier}: acoustic_only_reviewed must be boolean")
        clean_text(record.get("reviewer"), f"{identifier}.reviewer")
        clean_text(record.get("reviewed_at"), f"{identifier}.reviewed_at")
        if status == "approved":
            clean_text(final, f"{identifier}.caption_final", required=True)
            clean_text(record.get("reviewer"), f"{identifier}.reviewer", required=True)
            clean_text(record.get("reviewed_at"), f"{identifier}.reviewed_at", required=True)
            try:
                timestamp = datetime.fromisoformat(record["reviewed_at"].replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    raise ValueError("timezone missing")
            except ValueError as error:
                raise ValueError(f"{identifier}: reviewed_at must be ISO8601 with timezone") from error
            if not record["acoustic_only_reviewed"]:
                raise ValueError(f"{identifier}: human acoustic-only review required for B0")
        elif require_approved:
            raise ValueError(f"{identifier}: caption not approved ({status})")
        validate_perception(record.get("perception"), actual_identity)
    sites: dict[str, set[str]] = {}
    for row in rows:
        if row.get("site_id"):
            sites.setdefault(row["site_id"], set()).add(row["split"])
    return {"status": "READY_FOR_B0_CAPTION_EXPORT" if counts["approved"] == len(rows)
                      else "WAITING_FOR_CAPTION_REVIEW",
            "registry_version": registry_version, "rows": len(rows), "review_counts": counts,
            "tokenizer_validation": "UNVERIFIED_EXACT_PINNED_TOKENIZER_NOT_RUN",
            "acoustic_only_validation": "reviewer attestation; semantics not machine-verified",
            "split_policy": "source; site-held-out policy not enforced by this caption utility",
            "site_overlap_across_splits": {site: sorted(splits) for site, splits in sites.items() if len(splits) > 1},
            "missing_site_ids": sum(not row.get("site_id") for row in rows),
            "media_integrity": "manifest metadata only; media not opened or rehashed",
            "training_readiness": "NOT_ASSESSED: official prepare/preflight/extract/smoke still required",
            "media_files_read_or_changed": 0}


def validate_files(manifest: Path, registry_path: Path, registry_version: str,
                   *, require_approved: bool = False) -> dict:
    data, _, rows = read_manifest(manifest)
    _, registry = load_registry(registry_path)
    return validate_registry(registry, data, rows, registry_version, require_approved=require_approved)


def export_registry(manifest: Path, registry_path: Path, registry_version: str,
                    output_version: str, output: Path, *, dry_run: bool = False) -> dict:
    version(output_version)
    if output.exists() or output.is_symlink():
        raise ValueError(f"Refusing to overwrite export/feature directory: {output}")
    data, fields, rows = read_manifest(manifest)
    registry_data, registry = load_registry(registry_path)
    summary = validate_registry(registry, data, rows, registry_version, require_approved=True)
    by_id = {record["canonical_id"]: record for record in registry["records"]}
    export_fields = list(fields)
    for field in ("label", "caption"):
        if field not in export_fields:
            export_fields.append(field)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=export_fields)
    writer.writeheader()
    for row in rows:
        text = by_id[canonical_id(row)]["caption_final"].strip()
        writer.writerow({**row, "label": text, "caption": text})
    exported_data = buffer.getvalue().encode("utf-8")
    summary.update({"status": "PREVIEW" if dry_run else "B0_CAPTION_EXPORT_COMPLETE",
                    "output_version": output_version, "output": str(output.resolve()), "dry_run": dry_run,
                    "source_manifest_path": str(manifest.resolve()),
                    "source_registry_path": str(registry_path.resolve()),
                    "created_at": utc_now(), "script_sha256": digest(Path(__file__).read_bytes()),
                    "files_sha256": {"input_manifest_snapshot.csv": digest(data),
                                     "registry_snapshot.json": digest(registry_data),
                                     "captions_manifest.csv": digest(exported_data)},
                    "note": "B0 caption export only. Optional perception metadata is not injected. "
                            "Changing captions requires a new official data plan and text features."})
    if not dry_run:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in (("input_manifest_snapshot.csv", data),
                              ("registry_snapshot.json", registry_data),
                              ("captions_manifest.csv", exported_data)):
            with (output / name).open("xb") as handle:
                handle.write(content)
        # Completion marker last: absence means partial output, not an approved export.
        with (output / "summary.json").open("x", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "validate", "export"):
        sub = commands.add_parser(command)
        sub.add_argument("--manifest", type=Path, required=True)
        sub.add_argument("--registry-version", required=True)
        if command != "init":
            sub.add_argument("--registry", type=Path, required=True)
        if command != "validate":
            sub.add_argument("--output", type=Path, required=True)
            sub.add_argument("--dry-run", action="store_true")
        if command == "validate":
            sub.add_argument("--require-approved", action="store_true")
        if command == "export":
            sub.add_argument("--output-version", required=True)
    args = parser.parse_args()
    try:
        if args.command == "init":
            result = init_registry(args.manifest, args.registry_version, args.output, dry_run=args.dry_run)
        elif args.command == "validate":
            result = validate_files(args.manifest, args.registry, args.registry_version,
                                    require_approved=args.require_approved)
        else:
            result = export_registry(args.manifest, args.registry, args.registry_version,
                                     args.output_version, args.output, dry_run=args.dry_run)
    except (ValueError, OSError, UnicodeError) as error:
        print(json.dumps({"status": "BLOCKED", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
