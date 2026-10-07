"""Guarded video-only fine-tuning with the pinned, unmodified MMAudio training core.

The launcher owns data selection, validation logging, receipts and resume handling.
Model, optimizer, flow-matching loss and update operations remain in official Runner.
The default command is a CPU-only preflight. CUDA work requires --execute. Run a
separate --smoke experiment before a formal run, which requires its accepted receipt.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import random
import shlex
import signal
import sys
import tempfile
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fine_tune.official_common import (MODEL_SPECS, PINNED_COMMIT,
                                      activate_official_repo, atomic_json,
                                      load_plan, sha256_file, verify_official_repo)


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def checked_file(path: str | Path, digest: str, label: str) -> Path:
    path = Path(path).expanduser().resolve()
    if not path.is_file() or sha256_file(path) != digest:
        raise ValueError(f"{label}: missing file or SHA256 mismatch: {path}")
    return path


def read_rows(path: Path, delimiter: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def verify_extraction(path: Path, repo: Path) -> tuple[dict, dict]:
    """Only a new official extraction receipt can satisfy this boundary."""
    ready = read_json(path)
    if ready.get("schema") == "official_bundle_training_binding_v1":
        from fine_tune.official_feature_bundle import training_view
        return training_view(path, repo)
    if ready.get("schema_version") != 1 or ready.get("status") != "READY_FOR_TRAINING":
        raise ValueError("An official extraction READY_FOR_TRAINING receipt is required")
    if ready.get("official_commit") != PINNED_COMMIT:
        raise ValueError("Extraction used a different official commit")
    plan_path = checked_file(ready["plan_path"], ready["plan_sha256"], "Frozen plan")
    plan = load_plan(plan_path)
    if ready.get("model") not in MODEL_SPECS or ready["model"] != plan["model"]:
        raise ValueError("Feature/model mismatch")
    if ready.get("preprocessing_id") != plan["preprocessing_id"]:
        raise ValueError("Preprocessing provenance mismatch; legacy custom features are not accepted")
    if not ready.get("official_sources", {}).get("files"):
        raise ValueError("Extraction official-source provenance is missing")
    if ready["official_sources"].get("commit") != PINNED_COMMIT:
        raise ValueError("Extraction source commit mismatch")
    for relative, digest in ready["official_sources"]["files"].items():
        source = (repo / relative).resolve()
        if not source.is_relative_to(repo):
            raise ValueError("Official source path escapes repository")
        checked_file(source, digest, "Official extraction source")
    splits = ready.get("splits", {})
    if set(splits) != {"train", "val", "test"}:
        raise ValueError("Exactly train/val/test extraction outputs are required")
    all_ids: set[str] = set()
    directories: list[Path] = []
    for name, entry in splits.items():
        frozen = plan["splits"][name]
        if any(entry.get(key) != frozen.get(key) for key in ("count", "tsv_sha256", "manifest_sha256")):
            raise ValueError(f"{name}: extraction output differs from the frozen plan")
        tsv = checked_file(entry["tsv"], entry["tsv_sha256"], f"{name} TSV")
        manifest = checked_file(entry["manifest"], entry["manifest_sha256"], f"{name} manifest")
        rows = read_rows(tsv, "\t")
        records = read_rows(manifest, ",")
        ids = [row.get("id") for row in rows]
        if not ids or any(not item for item in ids) or len(ids) != len(set(ids)):
            raise ValueError(f"Empty/duplicate IDs in {name}")
        if len(ids) != entry["count"] or ids != [row.get("clip_id") for row in records]:
            raise ValueError(f"TSV/manifest row order or count mismatch in {name}")
        if any(row.get("label") != record.get("label") for row, record in zip(rows, records)):
            raise ValueError(f"TSV/manifest caption mismatch in {name}")
        if any(row.get("split") != name for row in records):
            raise ValueError(f"Manifest split mismatch in {name}")
        if all_ids.intersection(ids):
            raise ValueError("A clip ID appears in multiple splits")
        all_ids.update(ids)
        memmap = Path(entry["memmap_dir"]).resolve()
        if not memmap.is_dir():
            raise ValueError(f"Missing {name} memmap directory")
        if any(memmap == other or memmap.is_relative_to(other) or other.is_relative_to(memmap)
               for other in directories):
            raise ValueError("Split memmaps must be separate, non-nested directories")
        directories.append(memmap)
        checksums = entry.get("feature_checksums", {})
        actual = {str(item.relative_to(memmap)) for item in memmap.rglob("*") if item.is_file()}
        if not checksums or set(checksums) != actual:
            raise ValueError(f"{name}: feature receipt does not cover every memmap file")
        for relative, digest in checksums.items():
            member = (memmap / relative).resolve()
            if not member.is_relative_to(memmap):
                raise ValueError("Feature path escapes memmap directory")
            checked_file(member, digest, f"{name} features")
    return ready, plan


def official_weight_receipt(repo: Path, weights: Path, model: str) -> dict:
    """Read the upstream published checksum without importing torch or downloading."""
    filename = f"mmaudio_{model}.pth"
    tree = ast.parse((repo / "mmaudio/utils/download_utils.py").read_text())
    links = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "links"
                                               for t in node.targets):
            links = ast.literal_eval(node.value)
    entries = [item for item in links or [] if item["name"] == filename]
    if len(entries) != 1 or not weights.is_file():
        raise ValueError(f"Official pretrained weights are required: {filename}")
    md5 = hashlib.md5()  # upstream published checksum; SHA256 is recorded as well
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            md5.update(block)
    if md5.hexdigest() != entries[0]["md5"]:
        raise ValueError("Initialization weights do not match the official pretrained release")
    return {"path": str(weights), "sha256": sha256_file(weights),
            "official_md5": md5.hexdigest(), "official_filename": filename}


def verify_encoder_files(encoders: dict) -> None:
    if not encoders or not {"vae", "synchformer", "clip"}.issubset(encoders):
        raise ValueError("Extraction encoder provenance is incomplete")
    for name, entry in encoders.items():
        if "path" in entry:
            checked_file(entry["path"], entry["sha256"], name)
        elif "snapshot" in entry and entry.get("revision") and entry.get("files"):
            snapshot = Path(entry["snapshot"]).resolve()
            for relative, digest in entry["files"].items():
                # Hugging Face snapshots normally contain symlinks to the blob store.
                if Path(relative).is_absolute() or ".." in Path(relative).parts:
                    raise ValueError("Invalid encoder snapshot member")
                checked_file(snapshot / relative, digest, name)
        else:
            raise ValueError(f"Unverifiable encoder receipt: {name}")


def recipe_identity(args: argparse.Namespace, ready: dict, weights: dict) -> dict:
    """Run length/output path do not change the smoke acceptance identity."""
    identity = {"launcher_sha256": sha256_file(Path(__file__)),
            "shared_workflow_sha256": {name: sha256_file(Path(__file__).with_name(name))
                                       for name in ("official_common.py", "official_extract.py")},
            "official_commit": PINNED_COMMIT,
            "extraction_ready_sha256": sha256_file(Path(args.extraction_ready)),
            "plan_sha256": ready["plan_sha256"], "model": ready["model"],
            "preprocessing_id": ready["preprocessing_id"],
            "pretrained_sha256": weights["sha256"],
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "warmup_steps": args.warmup_steps, "seed": args.seed,
            "amp": not args.fp32, "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip, "lr_schedule": "constant",
            "normalization": "official_pretrained_checkpoint_latent_mean_std",
            "ema": {"sigma_rels": [0.05, 0.1], "update_every": 1, "start": 0}}
    if ready.get("schema_version") == 2:
        identity["portable_bundle_validator_sha256"] = sha256_file(Path(__file__).with_name("official_feature_bundle.py"))
        identity["checkpoint_roundtrip_required"] = True
    return identity


def verify_smoke(path: Path, identity: dict) -> dict:
    smoke = read_json(path)
    if smoke.get("status") != "PASSED" or smoke.get("recipe") != identity:
        raise ValueError("Smoke receipt is missing, failed, or belongs to another data/model/recipe")
    if not 20 <= smoke.get("completed_updates", 0) <= 100:
        raise ValueError("Accepted smoke must contain 20–100 completed updates")
    if not smoke.get("optimizer_covers_all_trainable") or not smoke.get("finite_losses_and_gradients"):
        raise ValueError("Smoke has not confirmed optimizer/gradient checks")
    if smoke.get("updated_tensor_fraction", 0) < 0.9 or not smoke.get("modalities_verified"):
        raise ValueError("Smoke has not confirmed full-backbone updates and all modalities")
    if identity.get("checkpoint_roundtrip_required") and not smoke.get("checkpoint_roundtrip_verified"):
        raise ValueError("Portable bundle training requires a checkpoint reload/resume smoke")
    return smoke


def preflight(args: argparse.Namespace) -> dict:
    repo = Path(args.official_repo).expanduser().resolve()
    provenance = verify_official_repo(repo)
    ready, _ = verify_extraction(Path(args.extraction_ready).resolve(), repo)
    if ready["official_sources"] != provenance:
        raise ValueError("Extraction and training official source receipts differ")
    for name in ("steps", "batch_size", "val_batch_size", "val_every", "save_every", "ema_every"):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.num_workers < 0 or args.warmup_steps < 1:
        raise ValueError("num_workers must be >=0 and warmup_steps >=1")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        raise ValueError("weight_decay must be finite and nonnegative")
    if not math.isfinite(args.gradient_clip) or args.gradient_clip <= 0:
        raise ValueError("gradient_clip must be finite and positive")
    if ready["splits"]["train"]["count"] < args.batch_size:
        raise ValueError("Training split is smaller than batch_size")
    if args.smoke and (not 20 <= args.steps <= 100 or (args.resume and not args.checkpoint_probe)):
        raise ValueError("Smoke needs 20–100 steps; resume is allowed only for a checkpoint probe")
    if args.checkpoint_probe and not args.smoke:
        raise ValueError("--checkpoint-probe is only for smoke validation")
    if args.stop_after is not None and not 0 < args.stop_after < args.steps:
        raise ValueError("--stop-after must be positive and less than target steps")
    if not args.smoke and args.steps % args.ema_every:
        raise ValueError("Formal steps must be divisible by ema_every for final-step EMA synthesis")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or int(os.environ.get("LOCAL_RANK", "0")) != 0:
        raise ValueError("This audited launcher supports one GPU only; do not invoke torchrun")
    weights = official_weight_receipt(repo, Path(args.weights).expanduser().resolve(), ready["model"])
    encoders = ready.get("encoder_hashes", {})
    verify_encoder_files(encoders)
    mode = MODEL_SPECS[ready["model"]]["mode"]
    if f"vocoder{mode}" not in encoders:
        raise ValueError(f"Missing {mode} vocoder provenance")
    empty = repo / "ext_weights/empty_string.pth"
    if not empty.is_file():
        raise ValueError("Official ext_weights/empty_string.pth is missing; run official preparation first")
    identity = recipe_identity(args, ready, weights)
    identity["empty_string_sha256"] = sha256_file(empty)
    run_dir = Path(args.run_dir).expanduser().resolve()
    if not args.smoke:
        if not args.smoke_report:
            raise ValueError("Formal run requires --smoke-report from a completed matching smoke run")
        verify_smoke(Path(args.smoke_report).resolve(), identity)
    checkpoint = None
    if args.resume:
        checkpoint = Path(args.resume).expanduser().resolve()
        if checkpoint.parent != run_dir or not (run_dir / "run_metadata.json").is_file():
            raise ValueError("Resume requires this run's own checkpoint and existing metadata")
        metadata = read_json(run_dir / "run_metadata.json")
        if metadata.get("recipe") != identity or metadata.get("target_updates") != args.steps:
            raise ValueError("Resume data/model/recipe or target step count differs")
        if metadata.get("runtime_policy") != runtime_policy(args):
            raise ValueError("Resume validation/checkpoint/EMA policy differs from this run")
        receipt = read_json(checkpoint.with_suffix(".json"))
        checked_file(checkpoint, receipt["sha256"], "Resume checkpoint")
        if receipt.get("recipe") != identity or receipt.get("completed_updates", 0) > args.steps:
            raise ValueError("Resume checkpoint is incompatible or already complete")
        if receipt.get("completed_updates") == args.steps and (run_dir / "COMPLETED.json").exists():
            raise ValueError("This run already finished; no resume is needed")
        if not (run_dir / "ema_ckpts").is_dir():
            raise ValueError("Resume requires this run's preserved EMA history directory")
    elif run_dir.exists():
        raise ValueError("Run directory already exists. Choose a new run-dir or explicitly --resume")
    return {"repo": str(repo), "run_dir": str(run_dir), "ready": ready,
            "official_provenance": provenance, "weights": weights, "recipe": identity,
            "empty_string_sha256": sha256_file(empty), "resume": str(checkpoint) if checkpoint else None}


def runtime_policy(args: argparse.Namespace) -> dict:
    return {"val_batch_size": args.val_batch_size, "val_every": args.val_every,
            "save_every": args.save_every, "ema_every": args.steps if args.smoke else args.ema_every,
            "smoke": args.smoke, "checkpoint_probe": args.checkpoint_probe, "num_workers": args.num_workers}


def execution_identity(torch):
    """Bind real GPU smoke/resume evidence to the actual destination environment."""
    packages = {}
    for name in ("torch", "torchvision", "torchaudio", "tensordict", "nitrous-ema",
                 "open-clip-torch", "hydra-core", "numpy"):
        packages[name] = importlib.metadata.version(name)
    return dict(python=sys.version.split()[0], packages=packages,
                gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                cudnn=torch.backends.cudnn.version(), tf32=True, cudnn_benchmark=False)


def equal_state(left, right, torch):
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict):
        return isinstance(right, dict) and left.keys() == right.keys() and all(
            equal_state(left[k], right[k], torch) for k in left)
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            equal_state(a, b, torch) for a, b in zip(left, right))
    return left == right


class RunLog:
    """Numerical logger; official Runner's logging calls remain supported."""
    def __init__(self, out: Path):
        self.events = (out / "official_metrics.jsonl").open("a", buffering=1)

    def info(self, value):
        print(value, flush=True)

    def warning(self, value):
        print(f"WARNING: {value}", flush=True)

    def log_string(self, key, value):
        self.info(f"{key}: {value}")

    def log_metrics(self, prefix, metrics, step, **kwargs):
        self.events.write(json.dumps({"utc": utc(), "kind": prefix, "step": step, **metrics},
                                     allow_nan=False) + "\n")

    def log_histogram(self, *args, **kwargs):
        pass

    def log_audio(self, *args, **kwargs):
        pass

    def log_spectrogram(self, *args, **kwargs):
        pass


def atomic_torch_save(torch, state: dict, target: Path) -> dict:
    temporary = target.with_suffix(target.suffix + ".pending")
    with temporary.open("wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, target)
    return {"path": str(target), "sha256": sha256_file(target), "bytes": target.stat().st_size}


def execute(args: argparse.Namespace, checked: dict) -> None:
    """An OS-held writer lock prevents concurrent resume into one run directory."""
    import fcntl
    out = Path(checked["run_dir"])
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
    # The lock file may remain after a process exits; flock itself is released by
    # the OS, so an old file is harmless and no manual stale-lock deletion is needed.
    with (out / ".training.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another training process already owns this run directory") from exc
        except OSError as exc:
            raise RuntimeError("Run filesystem does not support the required writer lock. "
                               "Use a local SSD run directory and separately back up checkpoints to Drive.") from exc
        status_path = out / "training_status.json"
        previous_status = read_json(status_path) if status_path.exists() else None
        try:
            _execute_locked(args, checked)
        except BaseException as exc:
            current_status = read_json(status_path) if status_path.exists() else None
            if current_status == previous_status:
                completed = (previous_status or {}).get("completed_updates", 0)
                if checked.get("resume"):
                    completed = read_json(Path(checked["resume"]).with_suffix(".json"))["completed_updates"]
                atomic_json(status_path, {"phase": "initialization_failed",
                            "error": str(exc), "utc": utc(), "completed_updates": completed})
            raise
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _execute_locked(args: argparse.Namespace, checked: dict) -> None:
    """Imports CUDA dependencies only after explicit execute and complete preflight."""
    repo, out = Path(checked["repo"]), Path(checked["run_dir"])
    activate_official_repo(repo)
    # All encoder snapshots must already exist, pinned in extraction provenance.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["LOCAL_RANK"] = "0"
    os.environ["WORLD_SIZE"] = "1"
    import numpy as np
    import torch
    import torch.distributed as dist
    from hydra import compose, initialize_config_dir
    from nitrous_ema import PostHocEMA
    from omegaconf import OmegaConf, open_dict
    from torch.utils.data import DataLoader, DistributedSampler, Subset
    from mmaudio.data.extracted_vgg import ExtractedVGG
    from mmaudio.runner import Runner
    from mmaudio.utils.synthesize_ema import synthesize_ema
    from fine_tune.official_extract import local_encoder_resolution

    if not torch.cuda.is_available():
        raise RuntimeError("Execute needs CUDA; no GPU work was started")
    if not args.fp32 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("GPU lacks BF16; explicitly select --fp32 and rerun matching smoke")
    runtime = execution_identity(torch)
    if args.resume and read_json(out / "run_metadata.json").get("execution_identity") != runtime:
        raise ValueError("Checkpoint resume hardware/software identity differs")
    if not args.smoke and read_json(Path(args.smoke_report)).get("execution_identity") != runtime:
        raise ValueError("Formal training requires a smoke on this hardware/software identity")
    os.chdir(repo)  # official Runner resolves ext_weights/empty_string.pth here
    with initialize_config_dir(version_base="1.3", config_dir=str(repo / "config")):
        cfg = compose(config_name="train_config")
    ready = checked["ready"]
    spec = MODEL_SPECS[ready["model"]]
    ema_every = args.steps if args.smoke else args.ema_every
    with open_dict(cfg):
        cfg.exp_id = out.name
        cfg.model = ready["model"]
        cfg.compile = False
        cfg.amp = not args.fp32
        cfg.enable_grad_scaler = False
        cfg.enable_email = False
        cfg.debug = False
        cfg.seed = args.seed
        cfg.batch_size = args.batch_size
        cfg.eval_batch_size = args.val_batch_size
        cfg.num_workers = args.num_workers
        cfg.weights = checked["weights"]["path"]
        cfg.checkpoint = checked["resume"]
        cfg.learning_rate = args.learning_rate
        cfg.linear_warmup_steps = args.warmup_steps
        cfg.lr_schedule = "constant"
        cfg.num_iterations = args.steps
        cfg.weight_decay = args.weight_decay
        cfg.clip_grad_norm = args.gradient_clip
        cfg.cudnn_benchmark = False
        cfg.data_dim.latent_seq_len = spec["latent_seq_len"]
        cfg.data_dim.clip_seq_len = 64
        cfg.data_dim.sync_seq_len = 192
        cfg.ema.enable = False  # create EMA only AFTER pretrained weights have been loaded
        cfg.ema.checkpoint_folder = str(out / "ema_ckpts")
        cfg.ema.checkpoint_every = ema_every
        cfg.ema.start = 0
        cfg.ema.update_every = 1
        cfg.log_text_interval = 50
        cfg.log_extra_interval = args.steps + 1
        cfg.save_weights_interval = args.steps + 1
        cfg.save_checkpoint_interval = args.steps + 1
        cfg.save_copy_iterations = []
        cfg.val_interval = args.val_every
        cfg.eval_interval = args.steps + 1
        cfg.save_eval_interval = args.steps + 1
        enc = ready["encoder_hashes"]
        cfg[f"vae_{spec['mode']}_ckpt"] = enc["vae"]["path"]
        cfg.synchformer_ckpt = enc["synchformer"]["path"]
        if spec["mode"] == "16k":
            cfg.bigvgan_vocoder_ckpt = enc["vocoder16k"]["path"]
        for name, cfg_name in (("train", "ExtractedVGG"), ("val", "ExtractedVGG_val"),
                               ("test", "ExtractedVGG_test")):
            cfg.data[cfg_name].tsv = ready["splits"][name]["tsv"]
            cfg.data[cfg_name].memmap_dir = ready["splits"][name]["memmap_dir"]
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(0)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = False
    # FileStore needs local filesystem locking; the run output may be on Drive.
    rendezvous_dir = tempfile.TemporaryDirectory(prefix="mmaudio-ddp-")
    rendezvous = Path(rendezvous_dir.name) / "init"
    try:
        dist.init_process_group("nccl", init_method=f"file://{rendezvous}", rank=0, world_size=1,
                                timeout=timedelta(hours=2))
    except BaseException:
        rendezvous_dir.cleanup()
        raise
    log = None
    start = time.monotonic()
    steps = 0
    status = {"phase": "initializing", "target_updates": args.steps, "completed_updates": 0,
              "smoke": args.smoke, "test_evaluation_started": False}

    def publish(phase: str, **fields):
        status.update(phase=phase, utc=utc(), elapsed_seconds=time.monotonic() - start, **fields)
        atomic_json(out / "training_status.json", status)

    try:
        datasets = {}
        for split in ("train", "val"):
            entry = ready["splits"][split]
            datasets[split] = ExtractedVGG(entry["tsv"], premade_mmap_dir=entry["memmap_dir"],
                                          data_dim=cfg.data_dim)
            for name in ("mean", "std", "clip_features", "sync_features", "text_features"):
                if getattr(datasets[split], name).shape[0] != entry["count"]:
                    raise ValueError(f"{split}/{name}: physical row count mismatch")
        sampler = DistributedSampler(datasets["train"], num_replicas=1, rank=0,
                                     shuffle=True, seed=args.seed)
        generator = torch.Generator().manual_seed(args.seed + 2)
        loader = DataLoader(datasets["train"], batch_size=args.batch_size, sampler=sampler,
                            drop_last=True, num_workers=args.num_workers, generator=generator)
        validation = datasets["val"]
        if args.smoke:
            validation = Subset(validation, range(min(32, len(validation))))
        val_loader = DataLoader(validation, batch_size=args.val_batch_size, shuffle=False,
                                drop_last=False, num_workers=args.num_workers,
                                generator=torch.Generator().manual_seed(args.seed + 3))
        log = RunLog(out)
        # Pretrained stats are deliberately retained; no misleading train-stat override.
        with local_encoder_resolution(ready["encoder_hashes"]):
            trainer = Runner(cfg, log=log, run_path=out, for_training=True)
        trainer.load_weights(checked["weights"]["path"])
        model = trainer.network.module
        with open_dict(cfg):
            cfg.ema.enable = True
        trainer.ema = PostHocEMA(model, sigma_rels=cfg.ema.sigma_rels,
                                 update_every=cfg.ema.update_every,
                                 checkpoint_every_num_steps=ema_every,
                                 checkpoint_folder=cfg.ema.checkpoint_folder,
                                 step_size_correction=True).cuda()
        trainer.ema_start = 0
        parameters = {name: parameter for name, parameter in model.named_parameters()
                      if parameter.requires_grad}
        expected = {id(parameter) for parameter in parameters.values()}
        optimized = [id(p) for group in trainer.optimizer.param_groups for p in group["params"]]
        if len(optimized) != len(set(optimized)) or set(optimized) != expected:
            raise RuntimeError("Optimizer does not cover every trainable backbone parameter exactly once")
        if any(id(p) in expected for p in trainer.features.parameters()):
            raise RuntimeError("External feature encoders accidentally entered the optimizer")
        initial = {name: parameter.detach().cpu().clone() for name, parameter in parameters.items()} if args.smoke else {}
        normalisation = {name: value.detach().cpu().tolist() for name, value in
                         model.state_dict().items() if name in ("latent_mean", "latent_std")}
        packages = {}
        for name in ("torch", "torchaudio", "torchvision", "tensordict", "nitrous-ema", "open-clip-torch"):
            try:
                packages[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                packages[name] = "unavailable"
        if not args.resume:
            atomic_json(out / "run_metadata.json", {"schema_version": 1, "created_utc": utc(),
                        "execution_identity": runtime,
                        "recipe": checked["recipe"], "target_updates": args.steps,
                        "runtime_policy": runtime_policy(args),
                        "official_provenance": checked["official_provenance"],
                        "weights": checked["weights"], "encoder_hashes": ready["encoder_hashes"],
                        "empty_string_sha256": checked["empty_string_sha256"],
                        "active_latent_statistics": normalisation, "packages": packages,
                        "gpu": torch.cuda.get_device_name(0), "batches_per_epoch": len(loader),
                        "trainable_parameters": sum(p.numel() for p in parameters.values()),
                        "trainable_tensors": len(parameters),
                        "training_scope": "official generation backbone; fixed external encoders",
                        "custom_orchestration": "single-GPU video-only loaders; guarded saves; fixed-RNG raw validation; one-based logging; no test evaluation",
                        "validation_clips": len(validation), "ema_initialized_after_pretrained": True})
            atomic_json(out / "effective_config.json", OmegaConf.to_container(cfg, resolve=True))
        validation_rng = torch.Generator(device="cuda").manual_seed(args.seed + 1).get_state()
        best_validation = None
        if args.resume:
            checkpoint = torch.load(checked["resume"], map_location="cpu", weights_only=True)
            model.load_state_dict(checkpoint["weights"], strict=True)
            trainer.optimizer.load_state_dict(checkpoint["optimizer"])
            trainer.scheduler.load_state_dict(checkpoint["scheduler"])
            trainer.ema.load_state_dict(checkpoint["ema"])
            steps = checkpoint["completed_updates"]
            trainer.rng.set_state(checkpoint["runner_rng"])
            torch.set_rng_state(checkpoint["torch_rng"])
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
            random.setstate(checkpoint["python_rng"])
            state = checkpoint["numpy_rng"]
            np.random.set_state((state[0], np.asarray(state[1], dtype="uint32"), state[2], state[3], state[4]))
            restored = {"weights": model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
                        "scheduler": trainer.scheduler.state_dict(), "ema": trainer.ema.state_dict(),
                        "runner_rng": trainer.rng.get_state(), "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate()}
            if not all(equal_state(value, checkpoint[key], torch) for key, value in restored.items()):
                raise ValueError("Checkpoint state did not restore exactly")
            numpy_state = np.random.get_state()
            if [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]] != checkpoint["numpy_rng"]:
                raise ValueError("NumPy RNG did not restore exactly")
            atomic_json(out / "checkpoint_reload.json", dict(status="PASSED", recipe=checked["recipe"],
                execution_identity=runtime, checkpoint_sha256=sha256_file(checked["resume"]),
                completed_updates=steps, state_keys=list(restored) + ["numpy_rng"],
                loader_epoch=steps // len(loader), loader_batch_offset=steps % len(loader)))
            saved_best = checkpoint.get("best_validation")
            if saved_best is not None:
                # A validation/save can finish after the last periodic resume
                # checkpoint. Keep that genuinely observed best model as well.
                best_validation = read_json(out / "best_validation.json")
                checked_file(best_validation["path"], best_validation["sha256"], "Best raw validation weights")
                if (not math.isfinite(best_validation["loss"])
                        or best_validation["loss"] > saved_best["loss"]):
                    raise ValueError("Best validation artifact no longer matches this run's recorded history")
            del checkpoint
        observed = {}
        accumulation = {"sum": 0.0, "count": 0}
        core_train, core_val = trainer.train_fn, trainer.val_fn

        def checked_train(*values):
            result = core_train(*values)
            loss = float(result[2].detach())
            if not math.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; optimizer update rejected")
            observed["loss"] = loss
            return result

        def checked_val(*values):
            result = core_val(*values)
            losses = result[0].detach().float()
            if not torch.isfinite(losses).all():
                raise FloatingPointError("Nonfinite validation loss")
            accumulation["sum"] += float(losses.sum())
            accumulation["count"] += losses.numel()
            return result

        trainer.train_fn, trainer.val_fn = checked_train, checked_val

        def guard_optimizer(*_):
            gradients = [p.grad.detach() for p in parameters.values() if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):
                raise FloatingPointError("Missing/nonfinite gradient; optimizer update rejected")
            observed["gradient_norm_after_clip"] = float(torch.stack([g.float().norm() for g in gradients]).norm())
        trainer.optimizer.register_step_pre_hook(guard_optimizer)
        stop = {"requested": False}
        def request_stop(*_):
            stop["requested"] = True
        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

        def save_checkpoint():
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError("Nonfinite model state; checkpoint write rejected")
            numpy_state = np.random.get_state()
            payload = {"completed_updates": steps, "weights": model.state_dict(),
                       "optimizer": trainer.optimizer.state_dict(),
                       "scheduler": trainer.scheduler.state_dict(), "ema": trainer.ema.state_dict(),
                       "best_validation": best_validation,
                       "runner_rng": trainer.rng.get_state(), "torch_rng": torch.get_rng_state(),
                       "cuda_rng": torch.cuda.get_rng_state_all(), "python_rng": random.getstate(),
                       "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]}
            # Two atomic slots retain the previous checkpoint if a write is interrupted.
            slot = (steps // args.save_every) % 2
            target = out / f"checkpoint_slot_{slot}.pth"
            receipt = atomic_torch_save(torch, payload, target)
            receipt.update(recipe=checked["recipe"], completed_updates=steps, utc=utc())
            atomic_json(target.with_suffix(".json"), receipt)
            atomic_json(out / "latest_checkpoint.json", receipt)

        def validate():
            nonlocal best_validation
            publish("validating", completed_updates=steps)
            saved_rng = trainer.rng.get_state()
            trainer.rng.set_state(validation_rng)
            accumulation.update(sum=0.0, count=0)
            trainer.val_integrator.reset_except_hooks()
            trainer.val_integrator.binned_tensors.clear()
            trainer.val_integrator.binned_tensor_indices.clear()
            try:
                for batch in val_loader:
                    trainer.validation_pass(batch, steps)
            finally:
                trainer.rng.set_state(saved_rng)
            if accumulation["count"] != len(validation):
                raise RuntimeError("Validation did not cover its declared subset")
            result = {"step": steps, "loss": accumulation["sum"] / accumulation["count"],
                      "clips": accumulation["count"], "weight_type": "raw", "utc": utc()}
            with (out / "validation_history.jsonl").open("a") as handle:
                handle.write(json.dumps(result, allow_nan=False) + "\n")
            if not args.smoke and (best_validation is None or result["loss"] < best_validation["loss"]):
                best_weights = atomic_torch_save(torch, model.state_dict(), out / "best_validation_raw_weights.pth")
                best_validation = {**result, **best_weights,
                                   "selection": "lowest observed fixed-RNG raw-model validation loss in this run",
                                   "scope": "all observed validation passes, including before an interruption; EMA not selected"}
                atomic_json(out / "best_validation.json", best_validation)
            trainer.val_integrator.reset_except_hooks()
            trainer.val_integrator.binned_tensors.clear()
            trainer.val_integrator.binned_tensor_indices.clear()
            publish("running", validation=result, completed_updates=steps, best_validation=best_validation)
            trainer.enter_train()

        trainer.log.data_timer.start()
        if not args.resume:
            validate()
        with (out / "training_steps.jsonl").open("a", buffering=1) as events:
            while steps < args.steps and not stop["requested"]:
                epoch, skip = divmod(steps, len(loader))
                sampler.set_epoch(epoch)
                generator.manual_seed(args.seed + 2 + epoch)
                for index, batch in enumerate(loader):
                    if index < skip:
                        continue
                    if stop["requested"] or (out / "STOP_REQUESTED").exists():
                        stop["requested"] = True
                        break
                    expected_keys = {"a_mean", "a_std", "clip_features", "sync_features", "text_features"}
                    if not expected_keys.issubset(batch) or not all(bool(torch.isfinite(batch[k]).all()) for k in expected_keys):
                        raise ValueError("Loader has missing/nonfinite conditioning modalities")
                    learning_rate = trainer.optimizer.param_groups[0]["lr"]
                    trainer.train_pass(batch, steps + 1)
                    steps += 1
                    if args.smoke and not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                        raise FloatingPointError("Nonfinite parameter after optimizer update")
                    events.write(json.dumps({"step": steps, "learning_rate": learning_rate, **observed},
                                            allow_nan=False) + "\n")
                    if steps % 10 == 0 or steps == 1:
                        publish("running", completed_updates=steps, epoch=epoch, **observed)
                        # Official Integrator retains its binned arrays; release logged history.
                        if steps % cfg.log_text_interval == 0:
                            trainer.train_integrator.binned_tensors.clear()
                            trainer.train_integrator.binned_tensor_indices.clear()
                    if steps % args.val_every == 0 or steps == args.steps:
                        validate()
                    if steps % args.save_every == 0 or steps == args.steps:
                        save_checkpoint()
                    if args.stop_after is not None and steps >= args.stop_after:
                        stop["requested"] = True
                        break
                    if steps >= args.steps:
                        break
        if stop["requested"]:
            save_checkpoint()
            publish("paused", completed_updates=steps)
            return
        if args.smoke:
            changed = [name for name, p in parameters.items() if not torch.equal(initial[name], p.detach().cpu())]
            unchanged = sorted(set(parameters) - set(changed))
            fraction = len(changed) / len(parameters)
            reload_path = out / "checkpoint_reload.json"
            roundtrip = False
            if args.checkpoint_probe and args.resume and reload_path.is_file():
                reload_check = read_json(reload_path)
                roundtrip = (reload_check.get("status") == "PASSED"
                             and reload_check.get("recipe") == checked["recipe"]
                             and reload_check.get("execution_identity") == runtime
                             and 0 < reload_check.get("completed_updates", 0) < steps)
            if args.checkpoint_probe and not roundtrip:
                raise ValueError("Checkpoint probe must stop, reload in a new process and advance")
            report = {"schema_version": 1, "status": "PASSED" if fraction >= 0.9 else "FAILED",
                      "execution_identity": runtime, "checkpoint_roundtrip_verified": roundtrip,
                      "checkpoint_reload_sha256": sha256_file(reload_path) if roundtrip else None,
                      "completed_updates": steps, "recipe": checked["recipe"],
                      "optimizer_covers_all_trainable": True, "finite_losses_and_gradients": True,
                      "modalities_verified": True, "updated_tensors": len(changed),
                      "updated_tensor_fraction": fraction, "unchanged_tensors": unchanged,
                      "validation_clips": len(validation), "full_validation": len(validation) == len(datasets["val"]),
                      "ema_initialized_after_pretrained": True, "utc": utc()}
            atomic_json(out / "smoke_report.json", report)
            if report["status"] != "PASSED":
                raise RuntimeError("Too few backbone tensors updated; inspect smoke_report.json")
        else:
            raw = atomic_torch_save(torch, model.state_dict(), out / "final_raw_weights.pth")
            ema = synthesize_ema(cfg, cfg.ema.default_output_sigma, step=steps)
            ema_receipt = atomic_torch_save(torch, ema, out / "final_ema_sigma_rel_0p05.pth")
            atomic_json(out / "COMPLETED.json", {"completed_updates": steps,
                        "raw_weights": raw, "ema_weights": ema_receipt,
                        "best_validation": best_validation,
                        "recipe": checked["recipe"], "test_evaluation_started": False, "utc": utc()})
        publish("complete", completed_updates=steps)
    except BaseException as exc:
        publish("failed", completed_updates=steps, error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        if log:
            log.events.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        rendezvous_dir.cleanup()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--extraction-ready", required=True)
    result.add_argument("--official-repo", required=True)
    result.add_argument("--weights", required=True, help="Explicit official pretrained .pth (checksum verified)")
    result.add_argument("--run-dir", required=True)
    result.add_argument("--steps", required=True, type=int)
    result.add_argument("--batch-size", type=int, default=2)
    result.add_argument("--val-batch-size", type=int, default=8)
    result.add_argument("--learning-rate", type=float, default=2e-5)
    result.add_argument("--warmup-steps", type=int, default=50)
    result.add_argument("--weight-decay", type=float, default=1e-6)
    result.add_argument("--gradient-clip", type=float, default=1.0)
    result.add_argument("--seed", type=int, default=14159265)
    result.add_argument("--num-workers", type=int, default=2)
    result.add_argument("--val-every", type=int, default=500)
    result.add_argument("--save-every", type=int, default=500)
    result.add_argument("--ema-every", type=int, default=500)
    result.add_argument("--fp32", action="store_true")
    result.add_argument("--smoke", action="store_true", help="Fresh 20–100-update acceptance run")
    result.add_argument("--smoke-report", help="Passed acceptance receipt required for formal training")
    result.add_argument("--resume", help="Explicit checkpoint from the same existing run")
    result.add_argument("--checkpoint-probe", action="store_true", help="Smoke must verify an interrupted checkpoint in a new process")
    result.add_argument("--stop-after", type=int, help="Save and pause after this completed update, preserving the target recipe")
    result.add_argument("--execute", action="store_true", help="Authorize CUDA training; otherwise dry-run only")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        checked = preflight(args)
        summary = {"status": "PREFLIGHT_PASSED", "gpu_started": False,
                   "model": checked["ready"]["model"], "run_dir": checked["run_dir"],
                   "split_counts": {k: v["count"] for k, v in checked["ready"]["splits"].items()},
                   "steps": args.steps, "smoke": args.smoke,
                   "approximate_epochs": args.steps / (checked["ready"]["splits"]["train"]["count"] // args.batch_size),
                   "test_evaluation": "Separate explicit evaluation; never automatic"}
        print(json.dumps(summary, indent=2))
        if args.execute:
            execute(args, checked)
        else:
            command = [sys.executable, str(Path(__file__).resolve()), *(argv or sys.argv[1:]), "--execute"]
            print("Reviewed command (not executed): " + shlex.join(command))
        return 0
    except (ValueError, FileNotFoundError, KeyError) as exc:
        print(f"PREFLIGHT BLOCKED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
