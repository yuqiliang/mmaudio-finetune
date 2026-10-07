"""Caption-independent, bounded official audio/video extraction.

Only scheduling, immutable provenance, media staging and row persistence are
ours. VGGSound.sample and FeaturesUtils forwards remain upstream. No text
encoder is called and no training-ready marker or training memmap is produced.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import tarfile
import time

from fine_tune import official_common as common
from fine_tune import official_extract as shared

MEDIA_KEYS = ("mean", "std", "clip_features", "sync_features")
ROW_KEYS = ("clip_id", "split", "split_row", "manifest_row", "source_recording_id",
            "source_start_seconds", "sha256", "byte_size")
SENTINEL = "TECHNICAL_MEDIA_ONLY_NOT_A_TRAINING_CAPTION"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def media_rows(accepted_rows):
    counts = Counter()
    result = []
    for index, row in enumerate(accepted_rows):
        split = row["split"]
        result.append(dict(clip_id=row["training_id"], split=split,
            split_row=counts[split], manifest_row=index,
            source_recording_id=f"{row['source_dataset_id']}::{row['source_file']}",
            source_start_seconds=row.get("source_start_seconds", ""),
            sha256=row["media_sha256"], byte_size=int(row["byte_size"]),
            path=row["absolute_path"]))
        counts[split] += 1
    return result


def media_identity(plan):
    """Locations and captions are not encoder inputs; content/order/split are."""
    return {key: plan[key] for key in (
        "model", "spec", "normalize_audio", "official_commit",
        "accepted_manifest_sha256", "official_audit_binding_sha256")} | {
        "rows": [{key: row[key] for key in ROW_KEYS} for row in plan["rows"]]}


def validate_plan(plan):
    require(plan.get("scope") == "official_media_only_v1", "Not a media-only plan")
    require(plan.get("training_ready") is False, "Media-only plan cannot authorize training")
    require(plan["official_commit"] == common.PINNED_COMMIT, "Wrong official revision")
    require(plan["model"] == "small_44k" and plan["spec"] == common.MODEL_SPECS["small_44k"],
            "This media plan uses the audited 44.1 kHz encoder contract")
    require(plan["normalize_audio"] == common.NORMALIZE_AUDIO, "Wrong split normalization")
    require(shared.fingerprint(media_identity(plan)) == plan["media_plan_id"], "Media plan changed")
    ids, hashes, groups, counts = set(), set(), {}, Counter()
    for i, row in enumerate(plan["rows"]):
        cid, split = row["clip_id"], row["split"]
        require(common.SAFE_ID.fullmatch(cid) and cid not in ids, "Unsafe/duplicate ID")
        require(split in common.SPLITS and row["manifest_row"] == i
                and row["split_row"] == counts[split], "Unstable row/split order")
        require(len(row["sha256"]) == 64 and all(c in "0123456789abcdef" for c in row["sha256"])
                and row["sha256"] not in hashes and row["byte_size"] > 0, "Bad/duplicate media hash")
        group = row["source_recording_id"]
        require(group and groups.get(group, split) == split, "Source group crosses splits")
        groups[group] = split
        counts[split] += 1
        ids.add(cid)
        hashes.add(row["sha256"])
    require(all(counts[s] >= 2 for s in common.SPLITS), "Need at least two clips in each split")
    require(dict(counts) == plan["counts"], "Count mismatch")
    return plan


def load_plan(path):
    return validate_plan(json.loads(Path(path).read_text()))


def prepare(args):
    """Bind existing technical acceptance, without relabelling human decisions."""
    require(common.sha256_file(args.accepted_manifest) == args.accepted_sha256,
            "Accepted-manifest SHA256 mismatch")
    rows = common.read_csv(args.accepted_manifest)
    binding = json.loads((args.audit_dir / "binding.json").read_text())
    binding_id = binding.pop("binding_sha256")
    require(hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest() == binding_id,
            "Decoder audit binding changed")
    require(binding["official_commit"] == common.PINNED_COMMIT and
            binding["sample_rate"] == 44100 and binding["audio_samples"] == 353280 and
            binding["fallback_decoder"] is False, "Not the official 44k audit")
    summary = json.loads((args.audit_dir / "summary.json").read_text())
    require(summary["status"] == "COMPLETE" and summary["full_dataset_checked"] is True and
            summary["binding_sha256"] == binding_id and
            summary["selected_clips"] == summary["checked_clips"] == summary["manifest_clips"] and
            summary["passed"] == len(rows) and
            summary["passed"] + summary["failed"] == summary["checked_clips"], "Incomplete audit")
    receipt_hashes = {}
    for row in rows:
        cid = row["training_id"]
        require(row["clapboard_final_status"] == "included" and
                row["official_decoder_validated"] == "true" and row["official_decoder_status"] == "pass",
                f"Clip not technically accepted: {cid}")
        token = hashlib.sha256(cid.encode()).hexdigest()
        path = args.audit_dir / "receipts" / (token + ".json")
        receipt = json.loads(path.read_text())
        require(receipt["training_id"] == cid and receipt["status"] == "pass" and
                receipt["binding_sha256"] == row["official_decoder_binding_sha256"] == binding_id and
                receipt["actual_media_sha256"] == receipt["expected_media_sha256"] == row["media_sha256"] and
                receipt["absolute_path"] == row["absolute_path"], f"Audit receipt mismatch: {cid}")
        for name, shape in (("audio", [353280]), ("clip_video", [64, 3, 384, 384]),
                            ("sync_video", [200, 3, 224, 224])):
            require(receipt["tensors"][name]["shape"] == shape and
                    receipt["tensors"][name]["all_finite"], f"Bad official decode: {cid}/{name}")
        receipt_hashes[cid] = common.sha256_file(path)
    plan = dict(schema_version=1, scope="official_media_only_v1", model="small_44k",
        spec=common.MODEL_SPECS["small_44k"], normalize_audio=common.NORMALIZE_AUDIO,
        official_commit=common.PINNED_COMMIT, accepted_manifest_sha256=args.accepted_sha256,
        official_audit_binding_sha256=binding_id, counts=dict(Counter(r["split"] for r in rows)),
        rows=media_rows(rows), training_ready=False, caption_status="UNCHANGED_NOT_APPROVED",
        content_review_status="UNCHANGED", created_at=now(),
        provenance=dict(accepted_manifest=str(args.accepted_manifest.resolve()),
            audit_dir=str(args.audit_dir.resolve()), receipt_hashes=receipt_hashes,
            summary_sha256=common.sha256_file(args.audit_dir / "summary.json")))
    plan["media_plan_id"] = shared.fingerprint(media_identity(plan))
    validate_plan(plan)
    require(not args.output.exists(), "Plan output exists; choose a new version")
    common.atomic_json(args.output, plan)
    print(json.dumps({"plan": str(args.output), "counts": plan["counts"],
                      "media_plan_id": plan["media_plan_id"], "training_ready": False}, indent=2))


def select_rows(plan, stage="probe6", batch_index=0, batch_size=500):
    rows = plan["rows"]
    first = []
    for split in common.SPLITS:
        members = [r for r in rows if r["split"] == split]
        if not members:
            continue
        one = members[0]
        two = next((r for r in members if r["source_recording_id"] != one["source_recording_id"]),
                   members[1] if len(members) > 1 else None)
        first.extend([one] + ([two] if two else []))
    selected = {r["clip_id"] for r in first}
    order = first + [r for r in rows if r["clip_id"] not in selected]
    if stage == "probe6":
        return first
    if stage == "pilot100":
        return order[:100]
    require(stage == "batch" and batch_index >= 0 and 1 <= batch_size <= 500,
            "Use a nonnegative batch index and batch size 1–500")
    chunk = order[batch_index * batch_size:(batch_index + 1) * batch_size]
    require(chunk, "Batch index outside this plan")
    return chunk


def source_path(row, media_root=None):
    return (Path(media_root) / row["split"] / (row["clip_id"] + ".mp4")
            if media_root else Path(row["path"]))


def verify_media(path, row):
    require(path.is_file() and path.stat().st_size == row["byte_size"] and
            common.sha256_file(path) == row["sha256"], f"Missing/changed official media: {path}")


def stage_media(args):
    plan = load_plan(args.plan)
    rows = select_rows(plan, args.stage, args.batch_index, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    with shared.exclusive_output(args.output):
        dst_plan = args.output / "media_plan.json"
        if dst_plan.exists():
            require(common.sha256_file(dst_plan) == common.sha256_file(args.plan), "Staged plan changed")
        for row in rows:
            src = source_path(row)
            dst = source_path(row, args.output / "media")
            verify_media(src, row)
            if dst.exists():
                verify_media(dst, row)
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".mp4.copying")
            shutil.copyfile(src, tmp)
            verify_media(tmp, row)
            os.replace(tmp, dst)
        if not dst_plan.exists():
            shutil.copyfile(args.plan, dst_plan)
        selection_id = shared.fingerprint([r["clip_id"] for r in rows])
        receipt = dict(status="STAGED_NOT_EXTRACTED", media_plan_id=plan["media_plan_id"],
            plan_sha256=common.sha256_file(dst_plan), stage=args.stage, batch_index=args.batch_index,
            batch_size=args.batch_size, selection_id=selection_id,
            selected=[{k: row[k] for k in ROW_KEYS} for row in rows], training_ready=False)
        common.atomic_json(args.output / f"staging_{args.stage}_{args.batch_index:03d}_{selection_id[:12]}.json", receipt)
        if args.archive:
            require(not args.archive.exists(), "Archive exists; refusing overwrite")
            temporary = args.archive.with_suffix(".tar.building")
            with tarfile.open(temporary, "w") as bundle:
                bundle.add(dst_plan, arcname="media_plan.json", recursive=False)
                for row in rows:
                    p = source_path(row, args.output / "media")
                    bundle.add(p, arcname=p.relative_to(args.output).as_posix(), recursive=False)
            with tarfile.open(temporary) as bundle:
                members = [bundle.getmember("media_plan.json")] + [bundle.getmember(
                    f"media/{r['split']}/{r['clip_id']}.mp4") for r in rows]
                for member, digest in zip(members, [receipt["plan_sha256"]] + [r["sha256"] for r in rows]):
                    require(hashlib.file_digest(bundle.extractfile(member), "sha256").hexdigest() == digest,
                            "Archive readback mismatch")
            os.replace(temporary, args.archive)
            common.atomic_json(args.archive.with_suffix(".tar.json"), dict(
                name=args.archive.name, bytes=args.archive.stat().st_size,
                sha256=common.sha256_file(args.archive), **receipt))
    print(f"Staged {len(rows)} verified media clips; no features extracted.")


def validate_media_features(features, spec):
    require(set(features) == set(MEDIA_KEYS), "Expected exactly four media tensor keys")
    for key in MEDIA_KEYS:
        tensor = features[key]
        require(tuple(tensor.shape) == shared.feature_shapes(spec)[key] and tensor.is_floating_point() and
                str(tensor.dtype) == "torch.float32", f"{key}: invalid shape/dtype")
        require(bool(tensor.isfinite().all()) and bool(tensor.ne(0).any()), f"{key}: nonfinite/zero feature")
    require(not bool(features["std"].lt(0).any()), "Negative posterior std")


def load_cached(cache, index, row, run_id, spec, torch):
    data_path, receipt_path = shared.row_paths(cache, index)
    r = json.loads(receipt_path.read_text())
    expected = dict(status="MEDIA_COMPLETE", run_id=run_id, row_index=index,
        clip_id=row["clip_id"], split=row["split"], manifest_row=row["manifest_row"],
        source_recording_id=row["source_recording_id"], source_sha256=row["sha256"])
    require(all(r.get(k) == v for k, v in expected.items()), "Media cache identity changed")
    require(common.sha256_file(data_path) == r["feature_sha256"], "Media cache hash changed")
    values = torch.load(data_path, map_location="cpu", weights_only=True)
    validate_media_features(values, spec)
    return values


def weight_identity(weights):
    return {k: {name: value for name, value in item.items() if name not in {"path", "snapshot"}}
            for k, item in weights.items()}


def require_previous_stage(output, plan, stage, run_id, spec, torch):
    previous = {"pilot100": "probe6", "batch": "pilot100"}.get(stage)
    if previous is None:
        return
    rows = select_rows(plan, previous)
    selection = shared.fingerprint([r["clip_id"] for r in rows])
    path = output / f"{previous}_000_{selection[:12]}_COMPLETE.json"
    require(path.is_file(), f"Run and verify {previous} before {stage}")
    report = json.loads(path.read_text())
    require(report.get("status") == "MEDIA_SELECTION_COMPLETE" and report.get("run_id") == run_id
            and report.get("selection_id") == selection and report.get("selected") == len(rows)
            and report.get("checked") == len(rows) and set(report.get("rows", [])) == {r["clip_id"] for r in rows},
            "Previous-stage completion does not match this run")
    for row in rows:
        load_cached(output / "row_cache" / row["split"], row["split_row"], row, run_id, spec, torch)


def extract(args):
    plan = load_plan(args.plan)
    chosen = select_rows(plan, args.stage, args.batch_index, args.batch_size)
    selection_id = shared.fingerprint([r["clip_id"] for r in chosen])
    require(args.clip_batch_size > 0 and args.sync_batch_size > 0, "Invalid encoder batch")
    official = common.verify_official_repo(args.official_repo)
    weights = shared.inventory_weights(args, plan["spec"])
    for row in chosen:
        verify_media(source_path(row, args.media_root), row)
    print(json.dumps(dict(mode="execute" if args.execute else "preflight_only",
        selected=len(chosen), stage=args.stage, counts=plan["counts"], training_ready=False)))
    if not args.execute:
        return
    common.activate_official_repo(args.official_repo)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    import torch
    from mmaudio.data.extraction.vgg_sound import VGGSound
    from mmaudio.model.utils.features_utils import FeaturesUtils
    require(torch.cuda.is_available(), "CUDA required; no preprocessing fallback")
    versions = shared.runtime_versions()
    identity = dict(media_plan_id=plan["media_plan_id"], official_sources=official,
        encoder_hashes=weight_identity(weights), versions=versions,
        code={Path(m.__file__).name: common.sha256_file(m.__file__) for m in (common, shared)},
        media_extractor_sha256=common.sha256_file(__file__),
        clip_batch_size=args.clip_batch_size, sync_batch_size=args.sync_batch_size,
        dtype="float32", tf32=True, modalities=list(MEDIA_KEYS),
        hardware=dict(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                      cudnn=torch.backends.cudnn.version()))
    run_id = shared.fingerprint(identity)
    output = args.output.resolve()
    with shared.exclusive_output(output):
        state = output / "media_state.json"
        if state.exists():
            saved = json.loads(state.read_text())
            require(args.resume and saved.get("identity") == identity and saved.get("run_id") == run_id,
                    "Resume requires identical media/code/weights/runtime/options")
        else:
            require(not any(p.name != ".extraction.lock" for p in output.iterdir()), "Nonempty output without state")
            common.atomic_json(state, dict(run_id=run_id, identity=identity, training_ready=False))
        started, completed, reused = time.monotonic(), [], 0
        spec = plan["spec"]
        def progress(status, error=None):
            report = dict(status=status, run_id=run_id, media_plan_id=plan["media_plan_id"],
                stage=args.stage, batch_index=args.batch_index, selected=len(chosen),
                batch_size=args.batch_size, selection_id=selection_id,
                checked=len(completed), reused=reused, rows=completed, training_ready=False,
                updated_at=now(), elapsed_seconds=time.monotonic() - started)
            if error:
                report["error"] = error
            common.atomic_json(output / "progress.json", report)
            return report
        try:
            require_previous_stage(output, plan, args.stage, run_id, spec, torch)
            remaining = []
            for row in chosen:
                cache = output / "row_cache" / row["split"]
                if shared.row_paths(cache, row["split_row"])[1].exists():
                    load_cached(cache, row["split_row"], row, run_id, spec, torch)
                    completed.append(row["clip_id"])
                    reused += 1
                else:
                    remaining.append(row)
            progress("RUNNING")
            if remaining:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                with shared.local_encoder_resolution(weights):
                    encoder = FeaturesUtils(tod_vae_ckpt=weights["vae"]["path"],
                        synchformer_ckpt=weights["synchformer"]["path"], enable_conditions=True,
                        mode=spec["mode"]).eval().cuda()
                datasets = {}
                grouped = defaultdict(list)
                for row in remaining:
                    grouped[(row["split"], source_path(row, args.media_root).parent)].append(row)
                for (split, root), members in grouped.items():
                    token = shared.fingerprint([split, str(root), [r["clip_id"] for r in members]])
                    tsv = output / "technical_inputs" / (token + ".tsv")
                    tsv.parent.mkdir(exist_ok=True)
                    with tsv.open("w", newline="") as stream:
                        writer = csv.writer(stream, delimiter="\t")
                        writer.writerow(["id", "label"])
                        writer.writerows([r["clip_id"], SENTINEL] for r in members)
                    ds = VGGSound(root, tsv_path=tsv, sample_rate=spec["sample_rate"], duration_sec=8.0,
                        audio_samples=spec["audio_samples"], normalize_audio=plan["normalize_audio"][split])
                    require(list(ds.videos) == [r["clip_id"] for r in members], "Official decoder omitted/reordered IDs")
                    for index, row in enumerate(members):
                        datasets[row["clip_id"]] = (ds, index)
                for row in remaining:
                    path = source_path(row, args.media_root)
                    verify_media(path, row)
                    ds, index = datasets[row["clip_id"]]
                    sample = ds.sample(index)
                    inputs = shared.validate_sample(sample, dict(clip_id=row["clip_id"], label=SENTINEL), spec)
                    with torch.inference_mode():
                        dist = encoder.encode_audio(sample["audio"].unsqueeze(0).cuda())
                        features = dict(mean=dist.mean.detach().cpu().transpose(1, 2)[0].contiguous(),
                            std=dist.std.detach().cpu().transpose(1, 2)[0].contiguous(),
                            clip_features=encoder.encode_video_with_clip(sample["clip_video"].unsqueeze(0).cuda(),
                                batch_size=args.clip_batch_size)[0].detach().cpu().contiguous(),
                            sync_features=encoder.encode_video_with_sync(sample["sync_video"].unsqueeze(0).cuda(),
                                batch_size=args.sync_batch_size)[0].detach().cpu().contiguous())
                    validate_media_features(features, spec)
                    verify_media(path, row)
                    cache = output / "row_cache" / row["split"]
                    cache.mkdir(parents=True, exist_ok=True)
                    data, receipt = shared.row_paths(cache, row["split_row"])
                    temporary = data.with_suffix(".pth.tmp")
                    torch.save(features, temporary)
                    check = torch.load(temporary, map_location="cpu", weights_only=True)
                    validate_media_features(check, spec)
                    require(all(torch.equal(features[k], check[k]) for k in MEDIA_KEYS), "Readback mismatch")
                    os.replace(temporary, data)
                    common.atomic_json(receipt, dict(status="MEDIA_COMPLETE", run_id=run_id,
                        row_index=row["split_row"], manifest_row=row["manifest_row"], clip_id=row["clip_id"],
                        split=row["split"], source_recording_id=row["source_recording_id"],
                        source_sha256=row["sha256"], feature_sha256=common.sha256_file(data),
                        input_checks=inputs, normalize_audio=plan["normalize_audio"][row["split"]],
                        text_features_extracted=False, training_ready=False, completed_at=now()))
                    completed.append(row["clip_id"])
                    progress("RUNNING")
                    print(f"{len(completed)}/{len(chosen)} MEDIA_COMPLETE {row['clip_id']}", flush=True)
                    del sample, features, check, dist
            for row in chosen:
                verify_media(source_path(row, args.media_root), row)
                load_cached(output / "row_cache" / row["split"], row["split_row"], row, run_id, spec, torch)
            require(load_plan(args.plan)["media_plan_id"] == plan["media_plan_id"], "Plan changed during extraction")
            require(common.verify_official_repo(args.official_repo) == official and
                    shared.inventory_weights(args, spec) == weights, "Code/weights changed during extraction")
            report = progress("MEDIA_SELECTION_COMPLETE")
            common.atomic_json(output / f"{args.stage}_{args.batch_index:03d}_{selection_id[:12]}_COMPLETE.json", report)
        except BaseException as error:
            progress("INCOMPLETE", f"{type(error).__name__}: {error}")
            raise


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--accepted-manifest", type=Path, required=True)
    prep.add_argument("--accepted-sha256", required=True)
    prep.add_argument("--audit-dir", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    for name in ("stage", "extract"):
        s = sub.add_parser(name)
        s.add_argument("--plan", type=Path, required=True)
        s.add_argument("--output", type=Path, required=True)
        s.add_argument("--stage", choices=("probe6", "pilot100", "batch"), default="probe6")
        s.add_argument("--batch-index", type=int, default=0)
        s.add_argument("--batch-size", type=int, default=500)
        if name == "stage":
            s.add_argument("--archive", type=Path)
            continue
        s.add_argument("--official-repo", type=Path, required=True)
        s.add_argument("--media-root", type=Path)
        s.add_argument("--weights-dir", type=Path, required=True)
        s.add_argument("--clip-snapshot", type=Path, required=True)
        s.add_argument("--clip-revision", required=True)
        s.add_argument("--vocoder-snapshot", type=Path, required=True)
        s.add_argument("--vocoder-revision", required=True)
        s.add_argument("--clip-batch-size", type=int, default=8)
        s.add_argument("--sync-batch-size", type=int, default=1)
        s.add_argument("--resume", action="store_true")
        s.add_argument("--execute", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    {"prepare": prepare, "stage": stage_media, "extract": extract}[args.command](args)


if __name__ == "__main__":
    main()
