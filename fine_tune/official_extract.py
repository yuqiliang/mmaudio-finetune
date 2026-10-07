"""Auditable, resumable extraction using the pinned upstream MMAudio encoders.

No custom video transforms are defined here.  The upstream VGGSound.sample
method is called directly (its __getitem__ silently catches failures).  Only
offline weight resolution, row caching, and bounded-memory writing are adapted.
Run ``python -m fine_tune.official_extract --help`` without ML dependencies.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
from typing import Any
from unittest.mock import patch

CLIP_REPO = "apple/DFN5B-CLIP-ViT-H-14-384"
VOCODER_REPO = "nvidia/bigvgan_v2_44khz_128band_512x"
FEATURE_KEYS = ("mean", "std", "clip_features", "sync_features", "text_features")


def _common():
    # Keep --help usable before installing the GPU environment.
    from fine_tune import official_common
    return official_common


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def feature_shapes(spec: dict[str, Any]) -> dict[str, tuple[int, ...]]:
    audio = (spec["latent_seq_len"], spec["latent_dim"])
    return {"mean": audio, "std": audio, "clip_features": (64, 1024),
            "sync_features": (192, 768), "text_features": (77, 1024)}


def fingerprint(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def snapshot_inventory(path: Path, revision: str, *, kind: str) -> dict[str, Any]:
    """Require an explicitly pinned local HF snapshot; never download at runtime.

    A snapshot basename must be its commit revision, matching the normal HF
    cache layout. File hashes, rather than the basename alone, bind resumes.
    """
    path = path.resolve()
    if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError(f"{kind} revision must be a full 40-character commit SHA")
    if not path.is_dir() or path.name != revision:
        raise ValueError(f"{kind} snapshot must be a local directory named {revision}")
    if kind == "clip":
        required = ("open_clip_config.json", "open_clip_pytorch_model.bin")
    else:
        required = ("config.json", "bigvgan_generator.pt")
    for name in required:
        file = path / name
        if not file.is_file() or file.stat().st_size == 0:
            raise ValueError(f"Missing {kind} snapshot file: {file}")
    config = json.loads((path / required[0]).read_text(encoding="utf-8"))
    if kind == "clip":
        model = config.get("model_cfg", {})
        # The official DFN5B *-384 repo's pinned config uses a native 378px
        # image size (27 patches of 14px). MMAudio itself feeds 384px frames;
        # that unmodified stride-14 convolution also produces 27x27 patches.
        # Do not infer the config dimension from the repository's display name,
        # or resize the official decoder output to make a local guard pass.
        vision = model.get("vision_cfg", {})
        expected_vision = dict(image_size=378, layers=32, width=1280,
                               head_width=80, patch_size=14)
        text = model.get("text_cfg", {})
        expected_text = dict(context_length=77, vocab_size=49408,
                             width=1024, heads=16, layers=24)
        if (model.get("embed_dim") != 1024 or model.get("quick_gelu") is not True
            or any(vision.get(k) != v for k, v in expected_vision.items())
            or any(text.get(k) != v for k, v in expected_text.items())):
            raise ValueError("CLIP snapshot is not the expected DFN5B ViT-H-14-384 architecture")
        expected_preprocess = dict(mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
            interpolation="bicubic", resize_mode="squash")
        preprocess = config.get("preprocess_cfg", {})
        if any(preprocess.get(k) != v for k, v in expected_preprocess.items()):
            raise ValueError("CLIP snapshot differs from upstream preprocess_cfg")
    return {"snapshot": str(path), "revision": revision,
            "repo_id": CLIP_REPO if kind == "clip" else VOCODER_REPO,
            "files": {p.relative_to(path).as_posix(): _common().sha256_file(p)
                      for p in sorted(path.rglob("*")) if p.is_file()}}


def inventory_weights(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    weights = args.weights_dir.resolve()
    names = {"vae": spec["vae_filename"], "synchformer": "synchformer_state_dict.pth"}
    if spec["vocoder_filename"]:
        names["vocoder16k"] = spec["vocoder_filename"]
    catalog = official_weight_catalog(args.official_repo)
    result = {}
    for key, filename in names.items():
        path = weights / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing local encoder weight: {path}")
        if filename not in catalog:
            raise ValueError(f"Encoder not listed in pinned official weight catalog: {filename}")
        md5 = hashlib.md5()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                md5.update(block)
        if md5.hexdigest() != catalog[filename]:
            raise ValueError(f"Encoder differs from published official weight: {path}")
        result[key] = {"path": str(path), "sha256": _common().sha256_file(path),
                       "official_md5": md5.hexdigest()}
    result["clip"] = snapshot_inventory(args.clip_snapshot, args.clip_revision, kind="clip")
    if spec["mode"] == "44k":
        if args.vocoder_snapshot is None or args.vocoder_revision is None:
            raise ValueError("44k extraction requires --vocoder-snapshot and --vocoder-revision")
        result["vocoder44k"] = snapshot_inventory(args.vocoder_snapshot, args.vocoder_revision, kind="vocoder")
    return result


def official_weight_catalog(repo: Path) -> dict[str, str]:
    """Read literal upstream published digests without importing/download side effects."""
    source = repo / "mmaudio/utils/download_utils.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == "links"
                                              for target in node.targets):
            links = ast.literal_eval(node.value)
            result = {item["name"]: item["md5"] for item in links}
            if not all(re.fullmatch(r"[0-9a-f]{32}", digest) for digest in result.values()):
                raise ValueError("Malformed official weight catalog")
            return result
    raise ValueError(f"No literal official weight catalog found in {source}")


def validate_manifest_sources(plan: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    """Prove official root/id.mp4 lookup reads the reviewed, hashed source file."""
    splits = {}
    seen = set()
    for split in ("train", "val", "test"):
        info = plan["splits"][split]
        rows = read_rows(Path(info["manifest"]))
        with Path(info["tsv"]).open(newline="", encoding="utf-8-sig") as stream:
            labels = list(csv.DictReader(stream, delimiter="\t"))
        if len(rows) != info["count"] or len(labels) != len(rows) or not rows:
            raise ValueError(f"{split}: manifest/TSV count mismatch or empty split")
        for row, label in zip(rows, labels):
            clip_id = row["clip_id"]
            if not clip_id or Path(clip_id).name != clip_id or clip_id in (".", "..") or clip_id in seen:
                raise ValueError(f"Unsafe or duplicate clip ID: {clip_id}")
            seen.add(clip_id)
            if label.get("id") != clip_id or label.get("label") != row["label"]:
                raise ValueError(f"{split}: manifest and TSV row order/caption mismatch at {clip_id}")
            if row.get("split") != split or row.get("manual_review_status") != "included" or row.get("qc_status") != "pass":
                raise ValueError(f"{clip_id}: manual-review/QC/split gate not satisfied")
            expected = Path(info["video_root"]) / (clip_id + ".mp4")
            source = Path(row["path"])
            if not source.is_file() or expected.resolve() != source.resolve():
                raise ValueError(f"{clip_id}: official root/id.mp4 does not resolve to reviewed source {source}")
            if _common().sha256_file(source) != row["sha256"]:
                raise ValueError(f"{clip_id}: source bytes changed since review/plan")
        splits[split] = rows
    return splits


def validate_sample(sample: dict[str, Any], row: dict[str, str], spec: dict[str, Any]) -> dict[str, Any]:
    if sample.get("id") != row["clip_id"] or sample.get("caption") != row["label"]:
        raise ValueError("Official dataset ID/caption differs from approved manifest")
    expected = {"audio": (spec["audio_samples"],), "clip_video": (64, 3, 384, 384),
                "sync_video": (200, 3, 224, 224)}
    report = {}
    for key, shape in expected.items():
        tensor = sample[key]
        if tuple(tensor.shape) != shape or not tensor.is_floating_point():
            raise ValueError(f"{row['clip_id']} {key}: shape/dtype mismatch, expected {shape}")
        if not bool(tensor.isfinite().all()):
            raise ValueError(f"{row['clip_id']} {key}: nonfinite input")
        low, high = float(tensor.min()), float(tensor.max())
        # Bicubic interpolation can overshoot, especially on high-contrast edges.
        # The signed transform is enforced by loading verified official source;
        # range alone cannot prove normalization (a bright frame can be > 0).
        bound = {"clip_video": (-0.25, 1.25), "sync_video": (-1.5, 1.5)}.get(key)
        if bound is not None and (low < bound[0] or high > bound[1]):
            raise ValueError(f"{row['clip_id']} {key}: values outside bicubic-tolerant bounds {bound}")
        report[key] = {"shape": list(shape), "dtype": str(tensor.dtype), "min": low, "max": high}
    return report


def validate_features(features: dict[str, Any], spec: dict[str, Any]) -> None:
    if set(features) != set(FEATURE_KEYS):
        raise ValueError("Feature row must contain exactly the five official modalities")
    for key, shape in feature_shapes(spec).items():
        value = features[key]
        if tuple(value.shape) != shape or not value.is_floating_point():
            raise ValueError(f"{key}: invalid feature shape/dtype, expected {shape}")
        if not bool(value.isfinite().all()):
            raise ValueError(f"{key}: nonfinite feature values")
        if not bool(value.ne(0).any()):
            raise ValueError(f"{key}: feature row is entirely zero")
    if bool(features["std"].lt(0).any()):
        raise ValueError("Audio posterior std contains negative values")


def row_paths(cache: Path, index: int) -> tuple[Path, Path]:
    return cache / f"{index:08d}.pth", cache / f"{index:08d}.json"


def row_receipt_valid(receipt: dict[str, Any], row: dict[str, str], index: int, run_id: str) -> bool:
    return (receipt.get("run_id") == run_id and receipt.get("row_index") == index
            and receipt.get("clip_id") == row["clip_id"] and receipt.get("label") == row["label"]
            and receipt.get("source_sha256") == row["sha256"] and receipt.get("status") == "COMPLETE")


def load_cached_row(cache: Path, index: int, row: dict[str, str], run_id: str,
                    spec: dict[str, Any], torch: Any) -> dict[str, Any]:
    data_path, receipt_path = row_paths(cache, index)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if not row_receipt_valid(receipt, row, index, run_id):
        raise ValueError(f"Cache identity changed: {receipt_path}")
    if _common().sha256_file(data_path) != receipt["feature_sha256"]:
        raise ValueError(f"Cache content changed: {data_path}")
    data = torch.load(data_path, map_location="cpu", weights_only=True)
    validate_features(data, spec)
    return data


@contextlib.contextmanager
def local_encoder_resolution(weights: dict[str, Any]):
    """Bind upstream constructors to hashed local artifacts, with no network IO.

    This does not alter any transform, model architecture, or encoder forward.
    It replaces open_clip's file resolver while leaving its upstream HF config
    parsing and model factory intact.
    """
    import open_clip.factory as factory
    from mmaudio.ext.autoencoder.autoencoder import BigVGANv2
    resolved_clip_files = set()

    def clip_file(model_id, filename=None, **kwargs):
        if model_id != CLIP_REPO:
            raise ValueError(f"Unexpected CLIP repository: {model_id}")
        filename = filename or "open_clip_pytorch_model.bin"
        if filename not in weights["clip"]["files"]:
            raise ValueError(f"Required file absent from pinned CLIP snapshot: {filename}")
        resolved_clip_files.add(filename)
        return str(Path(weights["clip"]["snapshot"]) / filename)

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(factory, "download_pretrained_from_hf", clip_file))
        if "vocoder44k" in weights:
            original = BigVGANv2.from_pretrained

            def local_vocoder(model_id, *args, **kwargs):
                if model_id != VOCODER_REPO:
                    raise ValueError(f"Unexpected vocoder repository: {model_id}")
                kwargs["local_files_only"] = True
                return original(weights["vocoder44k"]["snapshot"], *args, **kwargs)

            stack.enter_context(patch.object(BigVGANv2, "from_pretrained", local_vocoder))
        yield resolved_clip_files
        if not {"open_clip_config.json", "open_clip_pytorch_model.bin"}.issubset(resolved_clip_files):
            raise RuntimeError("Installed open_clip did not use the pinned local resolver; use supported open_clip_torch 2.29.0")


def runtime_versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0]}
    for package in ("torch", "torchvision", "torchaudio", "tensordict", "open_clip_torch", "huggingface_hub"):
        result[package] = importlib.metadata.version(package)
    return result


def extract_split(split: str, info: dict[str, Any], rows: list[dict[str, str]], *,
                  plan: dict[str, Any], spec: dict[str, Any], output: Path,
                  run_id: str, extractor: Any, torch: Any, dataset_class: Any,
                  clip_batch_size: int, sync_batch_size: int, limit: int | None = None) -> None:
    cache = output / "row_cache" / split
    cache.mkdir(parents=True, exist_ok=True)
    dataset = dataset_class(info["video_root"], tsv_path=info["tsv"],
                            sample_rate=spec["sample_rate"], duration_sec=8.0,
                            audio_samples=spec["audio_samples"],
                            normalize_audio=plan["normalize_audio"][split])
    if list(dataset.videos) != [row["clip_id"] for row in rows]:
        raise ValueError(f"{split}: official dataset silently omitted/reordered source IDs")
    work_rows = rows if limit is None else rows[:limit]
    for index, row in enumerate(work_rows):
        data_path, receipt_path = row_paths(cache, index)
        if receipt_path.exists():
            load_cached_row(cache, index, row, run_id, spec, torch)
            continue
        # A .pth without its receipt is an uncommitted interrupted write: it is
        # replaced only after successful extraction/readback of the same row.
        if _common().sha256_file(Path(row["path"])) != row["sha256"]:
            raise ValueError(f"Source changed while extracting: {row['clip_id']}")
        sample = dataset.sample(index)
        inputs = validate_sample(sample, row, spec)
        with torch.inference_mode():
            dist = extractor.encode_audio(sample["audio"].unsqueeze(0).cuda())
            features = {
                "mean": dist.mean.detach().cpu().transpose(1, 2)[0].contiguous(),
                "std": dist.std.detach().cpu().transpose(1, 2)[0].contiguous(),
                "clip_features": extractor.encode_video_with_clip(
                    sample["clip_video"].unsqueeze(0).cuda(), batch_size=clip_batch_size)[0].detach().cpu().contiguous(),
                "sync_features": extractor.encode_video_with_sync(
                    sample["sync_video"].unsqueeze(0).cuda(), batch_size=sync_batch_size)[0].detach().cpu().contiguous(),
                "text_features": extractor.encode_text([sample["caption"]])[0].detach().cpu().contiguous(),
            }
        validate_features(features, spec)
        if any(value.dtype != torch.float32 for value in features.values()):
            raise ValueError("Extraction must produce float32 features; mixed precision is not enabled")
        temporary = data_path.with_suffix(".pth.tmp")
        torch.save(features, temporary)
        check = torch.load(temporary, map_location="cpu", weights_only=True)
        validate_features(check, spec)
        if not all(torch.equal(check[key], features[key]) for key in FEATURE_KEYS):
            raise ValueError(f"Feature cache failed exact readback: {row['clip_id']}")
        os.replace(temporary, data_path)
        _common().atomic_json(receipt_path, {"status": "COMPLETE", "run_id": run_id,
            "row_index": index, "clip_id": row["clip_id"], "label": row["label"],
            "source_sha256": row["sha256"], "feature_sha256": _common().sha256_file(data_path),
            "input_checks": inputs})
        print(f"[{split}] {index + 1}/{len(work_rows)} {row['clip_id']}", flush=True)


def file_hashes(directory: Path) -> dict[str, str]:
    return {path.relative_to(directory).as_posix(): _common().sha256_file(path)
            for path in sorted(directory.rglob("*")) if path.is_file()}


@contextlib.contextmanager
def exclusive_output(directory: Path):
    """POSIX lock is released by the OS even after an interrupted Colab process."""
    import fcntl
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".extraction.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another extraction process is using {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def build_memmap(split: str, rows: list[dict[str, str]], info: dict[str, Any], *,
                 spec: dict[str, Any], output: Path, run_id: str, torch: Any) -> dict[str, Any]:
    """Allocate disk-backed tensors and stream a single checked row at a time."""
    import tensordict as td
    target = output / f"vgg-{split}"
    building = output / f"vgg-{split}.building"
    cache = output / "row_cache" / split
    if target.exists():
        mmap = td.TensorDict.load_memmap(target)
    elif building.exists():
        mmap = td.TensorDict.load_memmap(building)
    else:
        mmap = td.TensorDict({}, batch_size=[len(rows)]).memmap_(building)
    if list(mmap.batch_size) != [len(rows)]:
        raise ValueError(f"Partial memmap count differs: {building}")
    for key, shape in feature_shapes(spec).items():
        if key not in mmap.keys():
            if target.exists():
                raise ValueError(f"Published memmap missing modality: {target}/{key}")
            mmap.make_memmap(key, shape=(len(rows), *shape), dtype=torch.float32)
        elif tuple(mmap[key].shape) != (len(rows), *shape) or mmap[key].dtype != torch.float32:
            raise ValueError(f"Partial memmap shape/dtype differs: {key}")
    for index, row in enumerate(rows):
        values = load_cached_row(cache, index, row, run_id, spec, torch)
        for key in FEATURE_KEYS:
            if target.exists():
                if not torch.equal(mmap[key][index], values[key]):
                    raise ValueError(f"Published memmap differs from committed cache: {split} {index} {key}")
            else:
                mmap[key][index].copy_(values[key])
    del mmap
    if not target.exists():
        os.replace(building, target)
    # Re-open from its published path and verify every row and modality.
    reread = td.TensorDict.load_memmap(target)
    for index, row in enumerate(rows):
        values = load_cached_row(cache, index, row, run_id, spec, torch)
        if not all(torch.equal(reread[key][index], values[key]) for key in FEATURE_KEYS):
            raise ValueError(f"Memmap round-trip mismatch: {split} row {index}")
    del reread
    tsv, manifest = output / f"vgg-{split}.tsv", output / f"{split}_manifest.csv"
    for source, destination in ((Path(info["tsv"]), tsv), (Path(info["manifest"]), manifest)):
        if destination.exists() and _common().sha256_file(destination) != _common().sha256_file(source):
            raise ValueError(f"Output manifest already exists with different content: {destination}")
        if not destination.exists():
            shutil.copyfile(source, destination)
    return {"count": len(rows), "tsv": str(tsv), "tsv_sha256": _common().sha256_file(tsv),
            "memmap_dir": str(target), "manifest": str(manifest),
            "manifest_sha256": _common().sha256_file(manifest), "feature_checksums": file_hashes(target)}


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--plan", required=True, type=Path)
    result.add_argument("--official-repo", required=True, type=Path)
    result.add_argument("--weights-dir", required=True, type=Path)
    result.add_argument("--clip-snapshot", required=True, type=Path,
                        help="Local HF snapshot named by its commit SHA; config + open_clip_pytorch_model.bin")
    result.add_argument("--clip-revision", required=True, help="Exact 40-character HF commit SHA")
    result.add_argument("--vocoder-snapshot", type=Path, help="Required for 44k: local pinned BigVGAN snapshot")
    result.add_argument("--vocoder-revision", help="Required for 44k: exact BigVGAN HF commit SHA")
    result.add_argument("--output", required=True, type=Path)
    result.add_argument("--clip-batch-size", type=int, default=8, help="CLIP frames per forward (default 8)")
    result.add_argument("--sync-batch-size", type=int, default=1, help="Sync segments per forward (default 1)")
    result.add_argument("--probe-per-split", type=int,
                        help="With --execute: extract at most N first rows/split; publish PROBE_COMPLETE only; resume later without this option")
    result.add_argument("--execute", action="store_true", help="Authorize CUDA extraction; without this only validate/print")
    result.add_argument("--resume", action="store_true", help="Reuse exactly matching committed row caches")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    common = _common()
    if args.clip_batch_size < 1 or args.sync_batch_size < 1:
        raise ValueError("Encoder batch sizes must be positive")
    if args.probe_per_split is not None and args.probe_per_split < 1:
        raise ValueError("--probe-per-split must be at least 1")
    plan_path = args.plan.resolve()
    plan = common.load_plan(plan_path)
    spec = common.MODEL_SPECS[plan["model"]]
    official = common.verify_official_repo(args.official_repo.resolve())
    weights = inventory_weights(args, spec)
    rows = validate_manifest_sources(plan)
    output = args.output.resolve()
    print(json.dumps({"mode": "execute" if args.execute else "validation_only",
          "model": plan["model"], "dataset_version": plan["dataset_version"],
          "counts": {name: len(value) for name, value in rows.items()},
          "probe_per_split": args.probe_per_split,
          "output": str(output), "official": official}, indent=2))
    if not args.execute:
        return 0
    # Import only after explicit execution and preflight have succeeded.
    common.activate_official_repo(args.official_repo.resolve())
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from mmaudio.data.extraction.vgg_sound import VGGSound
    from mmaudio.model.utils.features_utils import FeaturesUtils
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA runtime is required; no CPU/MPS fallback changes are made")
    versions = runtime_versions()
    identity = {"schema_version": 1, "plan_sha256": common.sha256_file(plan_path),
                "official_sources": official, "encoder_hashes": weights,
                "extractor_sha256": common.sha256_file(Path(__file__)),
                "runtime_versions": versions, "clip_batch_size": args.clip_batch_size,
                "sync_batch_size": args.sync_batch_size}
    run_id = fingerprint(identity)
    with exclusive_output(output):
        return run_extraction(args, plan_path=plan_path, plan=plan, spec=spec, official=official,
                weights=weights, rows=rows, output=output, versions=versions, identity=identity,
                run_id=run_id, torch=torch, dataset_class=VGGSound, features_class=FeaturesUtils)


def run_extraction(args, *, plan_path, plan, spec, official, weights, rows, output,
                   versions, identity, run_id, torch, dataset_class, features_class):
    common = _common()
    state_path = output / "extraction_state.json"
    if any(path.name != ".extraction.lock" for path in output.iterdir()):
        if not args.resume or not state_path.exists():
            raise ValueError("Output is not empty; use a new output, or --resume with an identical extraction state")
        previous = json.loads(state_path.read_text())
        if previous.get("run_id") != run_id:
            raise ValueError("Resume refused: plan, source/code/encoder hashes, runtime, or options changed")
    else:
        output.mkdir(parents=True, exist_ok=True)
        common.atomic_json(state_path, {"run_id": run_id, "identity": identity})
    ready_path = output / "extraction_READY.json"
    if ready_path.exists():
        if args.probe_per_split is not None:
            raise ValueError("A probe cannot use an already completed extraction directory; choose a new output")
        ready = json.loads(ready_path.read_text())
        if (ready.get("status") != "READY_FOR_TRAINING" or ready.get("run_id") != run_id
                or ready.get("plan_sha256") != identity["plan_sha256"]
                or ready.get("encoder_hashes") != weights
                or ready.get("official_sources") != official
                or ready.get("model") != plan["model"]
                or ready.get("preprocessing_id") != plan["preprocessing_id"]
                or set(ready.get("splits", {})) != {"train", "val", "test"}):
            raise ValueError("Completed extraction receipt identity differs from the verified plan/run")
        for split, info in ready["splits"].items():
            if (info.get("count") != len(rows[split])
                    or info.get("tsv_sha256") != plan["splits"][split]["tsv_sha256"]
                    or info.get("manifest_sha256") != plan["splits"][split]["manifest_sha256"]):
                raise ValueError(f"Completed split identity differs: {split}")
            if file_hashes(Path(info["memmap_dir"])) != info["feature_checksums"]:
                raise ValueError(f"Completed memmap changed: {split}")
            for field in ("tsv", "manifest"):
                if common.sha256_file(Path(info[field])) != info[field + "_sha256"]:
                    raise ValueError(f"Completed {field} changed: {split}")
        print(f"Already complete and verified: {ready_path}")
        return 0
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        with local_encoder_resolution(weights):
            extractor = features_class(tod_vae_ckpt=weights["vae"]["path"],
                 bigvgan_vocoder_ckpt=weights.get("vocoder16k", {}).get("path"),
                 synchformer_ckpt=weights["synchformer"]["path"],
                 enable_conditions=True, mode=spec["mode"]).eval().cuda()
        completed = {}
        for split in ("train", "val", "test"):
            extract_split(split, plan["splits"][split], rows[split], plan=plan, spec=spec,
                          output=output, run_id=run_id, extractor=extractor, torch=torch,
                          dataset_class=dataset_class, clip_batch_size=args.clip_batch_size,
                          sync_batch_size=args.sync_batch_size, limit=args.probe_per_split)
            if args.probe_per_split is None:
                completed[split] = build_memmap(split, rows[split], plan["splits"][split],
                              spec=spec, output=output, run_id=run_id, torch=torch)
            else:
                completed[split] = {"verified_rows": min(args.probe_per_split, len(rows[split])),
                                    "full_count": len(rows[split])}
        # Revalidate all input hashes before publishing success, including clips
        # completed in an earlier invocation and the reviewed plan itself.
        common.load_plan(plan_path)
        if common.sha256_file(plan_path) != identity["plan_sha256"]:
            raise ValueError("Plan changed during extraction")
        validate_manifest_sources(plan)
        if inventory_weights(args, spec) != weights or common.verify_official_repo(args.official_repo.resolve()) != official:
            raise ValueError("Encoder weights or official source changed during extraction")
        if args.probe_per_split is not None:
            probe_path = output / "PROBE_COMPLETE.json"
            common.atomic_json(probe_path, {"schema_version": 1, "status": "PROBE_COMPLETE",
                "run_id": run_id, "plan_path": str(plan_path), "plan_sha256": identity["plan_sha256"],
                "model": plan["model"], "official_sources": official, "encoder_hashes": weights,
                "splits": completed, "training_ready": False,
                "next_step": "Resume with --execute --resume, omitting --probe-per-split, after reviewing probe receipts"})
            print(f"Probe completed; full dataset is not training-ready: {probe_path}")
            return 0
        common.atomic_json(ready_path, {"schema_version": 1, "status": "READY_FOR_TRAINING",
            "run_id": run_id, "plan_path": str(plan_path), "plan_sha256": identity["plan_sha256"],
            "dataset_version": plan["dataset_version"], "model": plan["model"],
            "official_commit": common.PINNED_COMMIT, "preprocessing_id": plan["preprocessing_id"],
            "official_sources": official, "encoder_hashes": weights,
            "runtime_versions": versions, "split_policy": plan["split_policy"],
            "normalize_audio": plan["normalize_audio"], "splits": completed})
        print(f"All rows and all splits verified: {ready_path}")
    except Exception as error:
        common.atomic_json(output / "extraction_failure.json",
                           {"run_id": run_id, "status": "INCOMPLETE", "error": str(error)})
        raise
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, FileNotFoundError) as error:
        raise SystemExit(f"ERROR: {error}")
