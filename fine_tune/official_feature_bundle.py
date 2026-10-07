"""Finish Colab features and verify a portable, provenance-bound training bundle.

All commands are read-only unless --execute is supplied. Media encoders are
never run here. Original media caches, plans and receipts are never rewritten.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from datetime import datetime

from fine_tune import official_common as common
from fine_tune import official_extract as shared
from fine_tune import official_media_extract as media

require = media.require
SCHEMA = "official_feature_bundle_v1"
REVIEW_FIELDS = (*media.ROW_KEYS, "caption_candidate", "caption_final",
                 "caption_review_status", "content_review_status", "reviewer",
                 "reviewed_at", "acoustic_only_reviewed")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def member(root, relative):
    path = PurePosixPath(relative)
    require(relative and not path.is_absolute() and ".." not in path.parts,
            "Unsafe bundle member")
    root = Path(root).resolve()
    result = root.joinpath(*path.parts)
    require(result.resolve().is_relative_to(root), "Bundle member escapes root")
    return result


def checked_json(path, digest):
    require(common.sha256_file(path) == digest, "JSON SHA256 mismatch")
    return read_json(path)


def csv_bytes(rows, fields, delimiter=","):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def media_contract(plan_path, state_path, complete_path, complete_sha256):
    plan = media.load_plan(plan_path)
    state = read_json(state_path)
    identity = state["identity"]
    require(shared.fingerprint(identity) == state["run_id"], "Media run identity changed")
    require(identity["media_plan_id"] == plan["media_plan_id"]
            and identity["official_sources"]["commit"] == common.PINNED_COMMIT
            and identity["dtype"] == "float32"
            and identity["modalities"] == list(media.MEDIA_KEYS), "Media identity contract differs")
    complete = checked_json(complete_path, complete_sha256)
    count = len(plan["rows"])
    batches = [media.select_rows(plan, "batch", i, 500) for i in range((count + 499) // 500)]
    ordered = [r["clip_id"] for batch in batches for r in batch]
    require(complete.get("status") == f"MEDIA_{count}_CPU_VERIFIED"
            and complete.get("checked") == count
            and complete.get("rows") == ordered and len(set(ordered)) == count
            and complete.get("batch_indices") == list(range(len(batches)))
            and complete.get("run_id") == state["run_id"]
            and complete.get("media_plan_id") == plan["media_plan_id"]
            and complete.get("training_ready") is False, "Media completion binding/coverage differs")
    return plan, state, complete


def review_template(plan, candidates):
    rows = common.read_csv(candidates)
    by_id = {r.get("training_id") or r.get("clip_id"): r for r in rows}
    require(len(by_id) == len(rows) == len(plan["rows"])
            and set(by_id) == {r["clip_id"] for r in plan["rows"]}, "Caption candidate coverage differs")
    result = []
    for row in plan["rows"]:
        source = by_id[row["clip_id"]]
        require(source["split"] == row["split"], "Caption candidate split differs")
        candidate = source.get("caption_qc_candidate") or source.get("caption_candidate")
        require(isinstance(candidate, str) and candidate.strip(), "Missing caption candidate")
        result.append({**{k: row[k] for k in media.ROW_KEYS}, "caption_candidate": candidate,
                       "caption_final": "", "caption_review_status": "pending",
                       "content_review_status": "pending", "reviewer": "", "reviewed_at": "",
                       "acoustic_only_reviewed": "false"})
    return result


def validate_review(plan, rows):
    require(len(rows) == len(plan["rows"]), "Review count differs")
    for row, source in zip(rows, plan["rows"]):
        require(all(str(row.get(k, "")) == str(source[k]) for k in media.ROW_KEYS),
                "Review clip/order/split/media identity differs")
        text = row.get("caption_final", "")
        require(isinstance(text, str) and text.strip() and not any(c in text for c in "\t\n\r"),
                "A nonempty single-line final caption is required")
        require(row.get("caption_review_status") == "approved"
                and row.get("content_review_status") == "approved"
                and str(row.get("acoustic_only_reviewed")).lower() == "true"
                and row.get("reviewer", "").strip(), "Caption/content human approval is pending")
        date = datetime.fromisoformat(row.get("reviewed_at", "").replace("Z", "+00:00"))
        require(date.tzinfo is not None, "Review timestamp needs timezone")


def frozen_captions(path, digest, plan):
    value = checked_json(path, digest)
    require(value.get("status") == "FROZEN_REVIEWED_CAPTIONS"
            and value.get("media_plan_id") == plan["media_plan_id"], "Caption freeze binding differs")
    validate_review(plan, value["rows"])
    return value


def validate_text(tensor, torch):
    require(tuple(tensor.shape) == (77, 1024) and tensor.dtype == torch.float32
            and bool(tensor.isfinite().all()) and bool(tensor.ne(0).any()), "Invalid text tensor")


def text_row(root, row, caption, run_id, torch):
    data, receipt = shared.row_paths(Path(root) / "row_cache" / row["split"], row["split_row"])
    value = read_json(receipt)
    require(value.get("status") == "TEXT_COMPLETE" and value.get("run_id") == run_id
            and value.get("clip_id") == row["clip_id"] and value.get("split") == row["split"]
            and value.get("row_index") == row["split_row"]
            and value.get("label") == caption["caption_final"], "Text cache binding differs")
    require(common.sha256_file(data) == value["feature_sha256"], "Text cache hash differs")
    tensor = torch.load(data, map_location="cpu", weights_only=True)
    validate_text(tensor, torch)
    return tensor, value


def text_identity(plan, state, frozen_hash, weights, official, torch):
    return dict(media_plan_id=plan["media_plan_id"], media_run_id=state["run_id"],
                frozen_sha256=frozen_hash, encoder_hashes=media.weight_identity(weights),
                official_sources=official, versions=shared.runtime_versions(),
                hardware=dict(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                              cudnn=torch.backends.cudnn.version()),
                code_sha256=common.sha256_file(__file__), dtype="float32", tf32=True,
                text_batch_size=1, tokenizer="ViT-H-14-378-quickgelu", context_length=77)


def extract_text(args, plan, state, frozen):
    official = common.verify_official_repo(args.official_repo)
    weights = shared.inventory_weights(args, plan["spec"])
    require(official == state["identity"]["official_sources"]
            and media.weight_identity(weights) == state["identity"]["encoder_hashes"],
            "Text and media must use identical official source and encoder weights")
    if not args.execute:
        return dict(status="TEXT_PREFLIGHT_ONLY", rows=len(plan["rows"]), gpu_started=False,
                    tokenizer_checked=False)
    common.activate_official_repo(args.official_repo)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from mmaudio.model.utils.features_utils import FeaturesUtils
    require(torch.cuda.is_available(), "Text extraction requires explicitly allocated CUDA")
    identity = text_identity(plan, state, args.frozen_sha256, weights, official, torch)
    run_id = shared.fingerprint(identity)
    output = args.output.resolve()
    require(not output.exists() or args.resume, "Text output exists; explicit resume required")
    with shared.exclusive_output(output):
        saved = output / "text_state.json"
        if saved.exists():
            require(read_json(saved) == dict(run_id=run_id, identity=identity), "Text resume identity differs")
        else:
            require(not any(p.name != ".extraction.lock" for p in output.iterdir()), "Unknown text output retained")
            common.atomic_json(saved, dict(run_id=run_id, identity=identity))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = False
        with shared.local_encoder_resolution(weights):
            encoder = FeaturesUtils(tod_vae_ckpt=None, synchformer_ckpt=weights["synchformer"]["path"],
                                    enable_conditions=True, mode="44k").eval().cuda()
        # Use the exact tokenizer's untruncated BPE length, including SOS/EOS.
        require(hasattr(encoder.tokenizer, "encode"), "Unsupported tokenizer; cannot prove non-truncation")
        lengths = [len(encoder.tokenizer.encode(r["caption_final"])) + 2 for r in frozen["rows"]]
        require(all(n <= 77 for n in lengths), "Caption would be truncated; review and refreeze, never trim silently")
        for row, caption in zip(plan["rows"], frozen["rows"]):
            data, receipt = shared.row_paths(output / "row_cache" / row["split"], row["split_row"])
            require(data.exists() == receipt.exists(), "Orphan text cache retained for inspection")
            require(not data.with_suffix(".pth.tmp").exists(), "Partial text cache retained for inspection")
            if data.exists():
                text_row(output, row, caption, run_id, torch)
                continue
            data.parent.mkdir(parents=True, exist_ok=True)
            with torch.inference_mode():
                tensor = encoder.encode_text([caption["caption_final"]])[0].detach().cpu().contiguous()
            validate_text(tensor, torch)
            temporary = data.with_suffix(".pth.tmp")
            torch.save(tensor, temporary)
            require(torch.equal(tensor, torch.load(temporary, map_location="cpu", weights_only=True)),
                    "Text serialization readback differs")
            os.replace(temporary, data)
            common.atomic_json(receipt, dict(status="TEXT_COMPLETE", run_id=run_id,
                clip_id=row["clip_id"], split=row["split"], row_index=row["split_row"],
                label=caption["caption_final"], feature_sha256=common.sha256_file(data)))
        frozen_captions(args.frozen, args.frozen_sha256, plan)
        for row, caption in zip(plan["rows"], frozen["rows"]):
            text_row(output, row, caption, run_id, torch)
        report = dict(status="TEXT_CPU_VERIFIED", run_id=run_id, checked=len(plan["rows"]),
                      frozen_sha256=args.frozen_sha256, max_tokens=max(lengths), truncations=0,
                      state_sha256=common.sha256_file(saved), training_ready=False)
        common.atomic_json(output / "TEXT_COMPLETE.json", report)
        return report


def verify_text_contract(root, plan, state, frozen_hash):
    text_state = read_json(root / "text_state.json")
    identity = text_state["identity"]
    complete = read_json(root / "TEXT_COMPLETE.json")
    require(shared.fingerprint(identity) == text_state["run_id"] == complete["run_id"]
            and identity["media_plan_id"] == plan["media_plan_id"]
            and identity["media_run_id"] == state["run_id"]
            and identity["frozen_sha256"] == complete["frozen_sha256"] == frozen_hash
            and identity["encoder_hashes"] == state["identity"]["encoder_hashes"]
            and identity["official_sources"] == state["identity"]["official_sources"]
            and complete.get("status") == "TEXT_CPU_VERIFIED"
            and complete.get("checked") == len(plan["rows"])
            and complete.get("truncations") == 0 and 2 <= complete["max_tokens"] <= 77
            and complete["state_sha256"] == common.sha256_file(root / "text_state.json"),
            "Text completion contract differs")
    return text_state, complete


def assemble(args, plan, state, frozen):
    text_state, _ = verify_text_contract(args.text_root, plan, state, args.frozen_sha256)
    require(not args.output.exists(), "Bundle output exists; use a fresh version, never overwrite")
    if not args.execute:
        return dict(status="ASSEMBLY_PREFLIGHT_ONLY", rows=len(plan["rows"]), training_ready=False)
    import torch
    import tensordict as td
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / "provenance"
    provenance.mkdir()
    for src, name in [(args.plan, "media_plan.json"), (args.state, "media_state.json"),
                      (args.complete, "MEDIA_COMPLETE.json"), (args.frozen, "captions_frozen.json"),
                      (args.text_root / "text_state.json", "text_state.json"),
                      (args.text_root / "TEXT_COMPLETE.json", "TEXT_COMPLETE.json")]:
        shutil.copyfile(src, provenance / name)
    labels = {r["clip_id"]: r for r in frozen["rows"]}
    splits = {}
    for split in common.SPLITS:
        rows = [r for r in plan["rows"] if r["split"] == split]
        directory = output / f"vgg-{split}"
        mmap = td.TensorDict({}, batch_size=[len(rows)]).memmap_(directory)
        for key, shape in shared.feature_shapes(plan["spec"]).items():
            mmap.make_memmap(key, shape=(len(rows), *shape), dtype=torch.float32)
        manifest = []
        for row in rows:
            index = row["split_row"]
            data, receipt = shared.row_paths(args.media_root / "row_cache" / split, index)
            record = read_json(receipt)
            require(record.get("normalize_audio") == plan["normalize_audio"][split]
                    and record.get("training_ready") is False
                    and record.get("text_features_extracted") is False, "Media normalization/scope differs")
            values = media.load_cached(data.parent, index, row, state["run_id"], plan["spec"], torch)
            text, text_receipt = text_row(args.text_root, row, labels[row["clip_id"]], text_state["run_id"], torch)
            values = {**values, "text_features": text}
            shared.validate_features(values, plan["spec"])
            for key, tensor in values.items():
                mmap[key][index].copy_(tensor)
            manifest.append(dict(clip_id=row["clip_id"], split=split, split_row=index,
                source_recording_id=row["source_recording_id"], sha256=row["sha256"],
                label=labels[row["clip_id"]]["caption_final"], media_run_id=state["run_id"],
                media_receipt_sha256=common.sha256_file(receipt), media_feature_sha256=record["feature_sha256"],
                text_run_id=text_state["run_id"], text_feature_sha256=text_receipt["feature_sha256"]))
        del mmap
        reread = td.TensorDict.load_memmap(directory)
        for row in rows:
            values = media.load_cached(args.media_root / "row_cache" / split, row["split_row"], row,
                                       state["run_id"], plan["spec"], torch)
            values["text_features"] = text_row(args.text_root, row, labels[row["clip_id"]],
                                               text_state["run_id"], torch)[0]
            require(all(torch.equal(reread[k][row["split_row"]], v) for k, v in values.items()),
                    "Bundle memmap exact readback differs")
        del reread
        tsv = output / f"vgg-{split}.tsv"
        tsv.write_bytes(csv_bytes([dict(id=r["clip_id"], label=r["label"]) for r in manifest], ("id", "label"), "\t"))
        csv_path = output / f"{split}_manifest.csv"
        csv_path.write_bytes(csv_bytes(manifest, tuple(manifest[0])))
        splits[split] = dict(count=len(rows), tsv=tsv.name, manifest=csv_path.name, memmap_dir=directory.name,
                             tsv_sha256=common.sha256_file(tsv), manifest_sha256=common.sha256_file(csv_path),
                             feature_checksums=shared.file_hashes(directory))
    media_contract(provenance / "media_plan.json", provenance / "media_state.json",
                   provenance / "MEDIA_COMPLETE.json", args.complete_sha256)
    frozen_captions(provenance / "captions_frozen.json", args.frozen_sha256, plan)
    receipt = dict(schema=SCHEMA, status="FEATURE_BUNDLE_READY", model=plan["model"],
        official_commit=common.PINNED_COMMIT, preprocessing_id=common.preprocessing_id(plan["model"]),
        media_run_id=state["run_id"], text_run_id=text_state["run_id"], media_plan_id=plan["media_plan_id"],
        media_complete_sha256=args.complete_sha256, frozen_sha256=args.frozen_sha256,
        encoder_hashes=state["identity"]["encoder_hashes"], official_sources=state["identity"]["official_sources"],
        splits=splits, files_sha256=shared.file_hashes(output),
        full_tensor_readback=True, training_started=False, smoke_required=True)
    common.atomic_json(output / "BUNDLE_READY.json", receipt)
    return dict(status=receipt["status"], ready=str(output / "BUNDLE_READY.json"),
                sha256=common.sha256_file(output / "BUNDLE_READY.json"), training_started=False)


def verify_bundle(path, expected_sha256):
    path = Path(path).resolve()
    ready = checked_json(path, expected_sha256)
    require(ready.get("schema") == SCHEMA and ready.get("status") == "FEATURE_BUNDLE_READY"
            and ready.get("full_tensor_readback") is True, "Not a verified official feature bundle")
    root = path.parent
    inventory = ready["files_sha256"]
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file() and p != path}
    require(set(inventory) == actual, "Bundle inventory has missing/unexpected files")
    for name, digest in inventory.items():
        require(common.sha256_file(member(root, name)) == digest, "Bundle file SHA256 mismatch: " + name)
    plan, state, _ = media_contract(root / "provenance/media_plan.json", root / "provenance/media_state.json",
        root / "provenance/MEDIA_COMPLETE.json", ready["media_complete_sha256"])
    frozen = frozen_captions(root / "provenance/captions_frozen.json", ready["frozen_sha256"], plan)
    text_state, _ = verify_text_contract(root / "provenance", plan, state, ready["frozen_sha256"])
    require(ready["model"] == plan["model"] and ready["media_plan_id"] == plan["media_plan_id"]
            and ready["official_commit"] == common.PINNED_COMMIT
            and ready["preprocessing_id"] == common.preprocessing_id(plan["model"])
            and ready["media_run_id"] == state["run_id"]
            and ready["text_run_id"] == text_state["run_id"]
            and ready["official_sources"] == state["identity"]["official_sources"]
            and ready["encoder_hashes"] == state["identity"]["encoder_hashes"], "Bundle lineage differs")
    captions = {r["clip_id"]: r["caption_final"] for r in frozen["rows"]}
    require(set(ready["splits"]) == set(common.SPLITS), "Exactly three splits required")
    for split, entry in ready["splits"].items():
        # Canonical layout also prevents split directories aliasing each other.
        require(entry["memmap_dir"] == f"vgg-{split}" and entry["tsv"] == f"vgg-{split}.tsv"
                and entry["manifest"] == f"{split}_manifest.csv", "Noncanonical bundle split layout")
        rows = common.read_csv(member(root, entry["manifest"]))
        original = [r for r in plan["rows"] if r["split"] == split]
        require(len(rows) == entry["count"] == len(original), "Bundle split count differs")
        for row, source in zip(rows, original):
            require(all(row[k] == str(source[k]) for k in ("clip_id", "split", "split_row", "source_recording_id", "sha256"))
                    and row["label"] == captions[source["clip_id"]]
                    and row["media_run_id"] == state["run_id"]
                    and row["text_run_id"] == text_state["run_id"], "Bundle row mapping differs")
        require(common.read_csv(member(root, entry["tsv"]), "\t") ==
                [dict(id=r["clip_id"], label=r["label"]) for r in rows], "Bundle TSV differs")
        require(shared.file_hashes(member(root, entry["memmap_dir"])) == entry["feature_checksums"],
                "Bundle tensor inventory differs")
        for key in ("tsv", "manifest"):
            require(inventory[entry[key]] == entry[key + "_sha256"], "Split metadata hash differs")
    return ready, plan


def bind_training(args):
    ready, plan = verify_bundle(args.bundle, args.bundle_sha256)
    official = common.verify_official_repo(args.official_repo)
    weights = shared.inventory_weights(args, plan["spec"])
    require(official == ready["official_sources"] and media.weight_identity(weights) == ready["encoder_hashes"],
            "Destination source/encoders differ from the Colab bundle")
    require(not args.output.exists(), "Runtime binding exists; choose a new path")
    result = dict(schema="official_bundle_training_binding_v1", status="READY_FOR_TRAINING_PREFLIGHT",
                  bundle_path=str(args.bundle.resolve()), bundle_sha256=args.bundle_sha256,
                  encoder_hashes=weights, training_started=False)
    if args.execute:
        common.atomic_json(args.output, result)
    return result


def training_view(path, repo):
    binding = read_json(path)
    require(binding.get("schema") == "official_bundle_training_binding_v1"
            and binding.get("status") == "READY_FOR_TRAINING_PREFLIGHT", "Invalid training binding")
    bundle = Path(binding["bundle_path"])
    ready, plan = verify_bundle(bundle, binding["bundle_sha256"])
    require(common.verify_official_repo(repo) == ready["official_sources"], "Training official source differs")
    require(media.weight_identity(binding["encoder_hashes"]) == ready["encoder_hashes"], "Bound encoder identity differs")
    splits = {name: {**entry, **{k: str(member(bundle.parent, entry[k])) for k in ("tsv", "manifest", "memmap_dir")}}
              for name, entry in ready["splits"].items()}
    return {**ready, "schema_version": 2, "plan_path": str(bundle.parent / "provenance/media_plan.json"),
            "plan_sha256": ready["files_sha256"]["provenance/media_plan.json"],
            "encoder_hashes": binding["encoder_hashes"], "splits": splits}, plan


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    for name in ("review-template", "freeze", "text", "assemble"):
        sub = commands.add_parser(name)
        sub.add_argument("--plan", type=Path, required=True)
        sub.add_argument("--output", type=Path, required=True)
        sub.add_argument("--execute", action="store_true")
        if name == "review-template":
            sub.add_argument("--candidates", type=Path, required=True)
        if name == "freeze":
            sub.add_argument("--review", type=Path, required=True)
        if name in ("text", "assemble"):
            sub.add_argument("--state", type=Path, required=True)
            sub.add_argument("--complete", type=Path, required=True)
            sub.add_argument("--complete-sha256", required=True)
            sub.add_argument("--frozen", type=Path, required=True)
            sub.add_argument("--frozen-sha256", required=True)
        if name == "text":
            add_weights(sub)
            sub.add_argument("--resume", action="store_true")
        if name == "assemble":
            sub.add_argument("--media-root", type=Path, required=True)
            sub.add_argument("--text-root", type=Path, required=True)
    sub = commands.add_parser("verify")
    sub.add_argument("--bundle", type=Path, required=True)
    sub.add_argument("--bundle-sha256", required=True)
    sub = commands.add_parser("bind-training")
    sub.add_argument("--bundle", type=Path, required=True)
    sub.add_argument("--bundle-sha256", required=True)
    sub.add_argument("--output", type=Path, required=True)
    sub.add_argument("--execute", action="store_true")
    add_weights(sub)
    return p


def add_weights(p):
    for name in ("official-repo", "weights-dir", "clip-snapshot", "vocoder-snapshot"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--clip-revision", required=True)
    p.add_argument("--vocoder-revision", required=True)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "verify":
        ready, plan = verify_bundle(args.bundle, args.bundle_sha256)
        result = dict(status="BUNDLE_HASHES_AND_BINDINGS_VERIFIED", counts=plan["counts"], gpu_started=False)
    elif args.command == "bind-training":
        result = bind_training(args)
    else:
        plan = media.load_plan(args.plan)
        if args.command == "review-template":
            rows = review_template(plan, args.candidates)
            require(not args.output.exists(), "Review output exists")
            if args.execute:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with args.output.open("xb") as f:
                    f.write(csv_bytes(rows, REVIEW_FIELDS))
            result = dict(status="PENDING_HUMAN_REVIEW", rows=len(rows), approved=0)
        elif args.command == "freeze":
            rows = common.read_csv(args.review)
            validate_review(plan, rows)
            require(not args.output.exists(), "Frozen output exists")
            result = dict(status="FROZEN_REVIEWED_CAPTIONS", media_plan_id=plan["media_plan_id"],
                          review_csv_sha256=common.sha256_file(args.review), rows=rows)
            if args.execute:
                common.atomic_json(args.output, result)
            result = dict(status="FROZEN_REVIEWED_CAPTIONS" if args.execute else "FREEZE_PREFLIGHT_ONLY", rows=len(rows))
        else:
            plan, state, _ = media_contract(args.plan, args.state, args.complete, args.complete_sha256)
            frozen = frozen_captions(args.frozen, args.frozen_sha256, plan)
            result = extract_text(args, plan, state, frozen) if args.command == "text" else assemble(args, plan, state, frozen)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
