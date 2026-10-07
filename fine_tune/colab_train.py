"""Prepare local Colab features and run bounded sessions with Drive recovery.

No stage writes files, installs packages, or starts GPU work without --execute.
Existing media encoders/caches and caption approval requirements are unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from fine_tune import official_common as common
from fine_tune import official_feature_bundle as bundle
from fine_tune.training_launch import weight_args, training_command, latest_checkpoint

require, read_json = bundle.require, bundle.read_json
STAGES = ("prepare", "smoke", "train", "resume")


def validate_config(config):
    require(config.get("schema") == "colab_training_config_v1", "Wrong Colab config schema")
    for key in ("code_root", "python", "official_repo", "weights_dir", "clip_snapshot", "vocoder_snapshot",
                "pretrained", "drive_mount", "drive_bundle_ready", "bundle_ready", "binding", "smoke_run", "train_run",
                "smoke_backup", "train_backup"):
        value = config.get(key)
        require(isinstance(value, str) and Path(value).is_absolute() and "<" not in value
                and "\n" not in value and "\r" not in value, "Set an absolute path: " + key)
    drive = Path(config["drive_mount"]).resolve()
    require(drive.is_relative_to("/content") and drive != Path("/content"), "Use a mounted Drive folder under /content")
    for key in ("drive_bundle_ready", "smoke_backup", "train_backup"):
        require(Path(config[key]).resolve().is_relative_to(drive), "Durable artifact must be on Drive: " + key)
    for key in ("code_root", "official_repo", "weights_dir", "clip_snapshot", "vocoder_snapshot", "pretrained",
                "bundle_ready", "binding", "smoke_run", "train_run"):
        path = Path(config[key]).resolve()
        require(path.is_relative_to("/content") and not path.is_relative_to(drive), "Use local /content for active data/code: " + key)
    roots = [Path(config[k]).resolve() for k in ("smoke_run", "train_run", "smoke_backup", "train_backup")]
    roots += [Path(config[k]).resolve().parent for k in ("drive_bundle_ready", "bundle_ready")]
    for i, path in enumerate(roots):
        require(not any(path.is_relative_to(other) or other.is_relative_to(path) for other in roots[:i]),
                "Bundle, run and backup directories must not overlap")
    require(not any(Path(config["binding"]).resolve().is_relative_to(p) for p in roots), "Binding must be outside runs/bundles")
    require(bool(re.fullmatch(r"[0-9a-f]{64}", config.get("bundle_sha256") or "")), "Bind exact BUNDLE_READY SHA256")
    require(type(config.get("session_updates")) is int and config["session_updates"] > 0, "Set a positive session update limit")
    r = config["recipe"]
    require(type(r["batch_size"]) is int and r["batch_size"] > 0 and r["precision"] in ("bf16", "fp32"), "Invalid recipe")
    return config


def colab_environment(config):
    require(sys.platform == "linux" and Path("/content").is_dir(), "Execute this workflow inside Colab")
    require(os.path.ismount(config["drive_mount"]), "Google Drive must be mounted before execution")
    require(Path(sys.executable).resolve() == Path(config["python"]).resolve(), "Use the configured Python environment")


def stage_bundle(config):
    source = Path(config["drive_bundle_ready"])
    target = Path(config["bundle_ready"])
    require(source.name == target.name == "BUNDLE_READY.json", "Use the canonical bundle receipt name")
    if target.parent.exists():
        bundle.verify_bundle(target, config["bundle_sha256"])
        return
    bundle.verify_bundle(source, config["bundle_sha256"])
    paths = list(source.parent.rglob("*"))
    require(not any(p.is_symlink() for p in paths), "Stage a self-contained bundle without symlinks")
    size = sum(p.stat().st_size for p in paths if p.is_file())
    ancestor = target.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    require(shutil.disk_usage(ancestor).free > size + 2 * 1024 ** 3, "Insufficient local space for bundle plus staging reserve")
    shutil.copytree(source.parent, target.parent)
    bundle.verify_bundle(target, config["bundle_sha256"])


def commands(config, stage):
    if stage == "prepare":
        return [[config["python"], "-m", "fine_tune.official_preflight", "--official-repo", config["official_repo"]],
                [config["python"], "-m", "fine_tune.official_feature_bundle", "bind-training",
                 "--bundle", config["bundle_ready"], "--bundle-sha256", config["bundle_sha256"],
                 "--output", config["binding"], *weight_args(config)]]
    command = training_command(config, stage)
    command += ["--backup-dir", config["smoke_backup"] if stage == "smoke" else config["train_backup"]]
    if stage == "smoke":
        return [command + ["--stop-after", "20"], command + ["--resume", "<verified step-20 checkpoint>"]]
    start = 0
    if stage == "resume":
        checkpoint = latest_checkpoint(config["train_run"])
        start = read_json(Path(checkpoint).with_suffix(".json"))["completed_updates"]
        command += ["--resume", checkpoint]
    stop = min(start + config["session_updates"], config["recipe"]["steps"])
    if stop < config["recipe"]["steps"]:
        command += ["--stop-after", str(stop)]
    return [command]


def run(config_path, stage, execute=False):
    config = validate_config(read_json(config_path))
    planned = commands(config, stage)
    if not execute:
        return dict(status="COLAB_PREVIEW", commands=[shlex.join(c) for c in planned],
                    gpu_started=False, automatic_next_session=False)
    colab_environment(config)
    env = {**os.environ, "PYTHONPATH": config["code_root"], "PYTHONNOUSERSITE": "1",
           "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "MPLBACKEND": "Agg"}

    def invoke(command, write=False):
        subprocess.run(command + (["--execute"] if write else []), check=True, cwd=config["code_root"], env=env)

    if stage == "prepare":
        invoke(planned[0])
        stage_bundle(config)
        if Path(config["binding"]).exists():
            binding = read_json(config["binding"])
            require(binding["bundle_path"] == str(Path(config["bundle_ready"]).resolve())
                    and binding["bundle_sha256"] == config["bundle_sha256"], "Existing binding targets another bundle")
            ready, _ = bundle.training_view(Path(config["binding"]), Path(config["official_repo"]))
            from fine_tune.official_train import verify_encoder_files
            verify_encoder_files(ready["encoder_hashes"])
        else:
            invoke(planned[1], True)
        return dict(status="COLAB_INPUTS_VERIFIED", training_started=False)
    require(not (Path(config["smoke_run"] if stage == "smoke" else config["train_run"]) / "STOP_REQUESTED").exists(),
            "A stop request exists; inspect it before explicitly continuing")
    if stage == "smoke":
        invoke(planned[0], True)
        state = read_json(Path(config["smoke_run"]) / "training_status.json")
        require(state["phase"] == "paused" and state["completed_updates"] == 20, "Smoke did not pause at step 20")
        from fine_tune.run_snapshot import latest, verify, restore
        recovery_path, recovery_sha = latest(config["smoke_backup"])
        recovery, _ = verify(recovery_path, recovery_sha)
        require(recovery["completed_updates"] == 20, "Drive recovery snapshot is not at step 20")
        local_run = Path(config["smoke_run"])
        retained = local_run.with_name(local_run.name + ".before-restore")
        require(not retained.exists(), "Previous smoke recovery copy exists; inspect before retry")
        local_run.rename(retained)
        restore(recovery_path, recovery_sha, local_run, True)
        planned[1][-1] = latest_checkpoint(config["smoke_run"])
        invoke(planned[1], True)
        result = read_json(Path(config["smoke_run"]) / "smoke_report.json")
        require(result["status"] == "PASSED" and result["checkpoint_roundtrip_verified"], "Smoke failed")
        common.atomic_json(Path(config["smoke_backup"]) / "COLAB_SMOKE_READY.json", dict(
            status="DRIVE_RESTORE_SMOKE_PASSED", restored_snapshot=str(recovery_path), restored_snapshot_sha256=recovery_sha,
            smoke_report_sha256=common.sha256_file(Path(config["smoke_run"]) / "smoke_report.json")))
    else:
        acceptance = read_json(Path(config["smoke_backup"]) / "COLAB_SMOKE_READY.json")
        require(acceptance.get("status") == "DRIVE_RESTORE_SMOKE_PASSED"
                and acceptance["smoke_report_sha256"] == common.sha256_file(Path(config["smoke_run"]) / "smoke_report.json"),
                "Colab training requires the matching Drive-restore smoke")
        invoke(planned[0], True)
    from fine_tune.run_snapshot import latest, verify
    backup = config["smoke_backup"] if stage == "smoke" else config["train_backup"]
    path, digest = latest(backup)
    receipt, _ = verify(path, digest)
    state = read_json(Path(config["smoke_run"] if stage == "smoke" else config["train_run"]) / "training_status.json")
    require(receipt["completed_updates"] == state["completed_updates"], "Durable snapshot is behind this session")
    return dict(status="COLAB_SESSION_SAVED", phase=state["phase"], completed_updates=state["completed_updates"],
                snapshot=str(path), snapshot_sha256=digest, automatic_next_session=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(run(args.config, args.stage, args.execute), indent=2))


if __name__ == "__main__":
    main()
