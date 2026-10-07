"""Render or run explicitly selected Myriad preparation/training stages.

Rendering does not connect, install, copy, submit, or run CUDA. Executing a stage
requires an SGE allocation, a reviewed local config, and an explicit flag.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from fine_tune import official_common as common
from fine_tune.official_feature_bundle import read_json, require
from fine_tune.training_launch import weight_args, training_command, latest_checkpoint

STAGES = ("verify", "smoke", "train", "resume")


def validate_config(config, *, concrete=False):
    require(config.get("schema") == "myriad_training_config_v1", "Wrong Myriad config schema")
    require(config.get("scheduler") == "sge", "Recheck current Myriad scheduler; this renderer supports SGE")
    resources = config["resources"]
    require(type(resources["cores"]) is int and 1 <= resources["cores"] <= 36, "Invalid CPU allocation")
    import re
    for key in ("mem_per_core", "tmpfs"):
        require(bool(re.fullmatch(r"[1-9][0-9]*[MGT]", resources[key])), "Invalid resource quantity")
    require(bool(re.fullmatch(r"[0-9]{1,2}:[0-5][0-9]:[0-5][0-9]", resources["walltime"])), "Invalid walltime")
    h, m, s = map(int, resources["walltime"].split(":"))
    require(0 < h * 3600 + m * 60 + s <= 48 * 3600, "Use a bounded job of at most 48 hours")
    require(resources.get("gpu_count") == 1, "This workflow is single-GPU")
    require(isinstance(config.get("modules"), list) and all(isinstance(x, str) and x for x in config["modules"]), "Module names must be explicit strings")
    paths = ("code_root", "python", "official_repo", "weights_dir", "clip_snapshot", "vocoder_snapshot",
             "pretrained", "bundle_ready", "binding", "smoke_run", "train_run", "job_output")
    if concrete:
        for key in paths:
            value = config.get(key)
            require(isinstance(value, str) and Path(value).is_absolute() and "<" not in value
                    and "\n" not in value and "\r" not in value, "Set an absolute path: " + key)
        require(config["smoke_run"] != config["train_run"], "Smoke and training must use separate run directories")
        require(bool(re.fullmatch(r"[0-9a-f]{64}", config.get("bundle_sha256", ""))), "Bind exact BUNDLE_READY SHA256")
        bundle_root = Path(config["bundle_ready"]).resolve().parent
        for key in ("binding", "smoke_run", "train_run", "job_output"):
            require(not Path(config[key]).resolve().is_relative_to(bundle_root), "Runtime outputs must stay outside the immutable bundle")
        smoke, train = (Path(config[k]).resolve() for k in ("smoke_run", "train_run"))
        require(not smoke.is_relative_to(train) and not train.is_relative_to(smoke), "Run directories must not overlap")
        for key in ("binding", "job_output"):
            require(not any(Path(config[key]).resolve().is_relative_to(p) for p in (smoke, train)),
                    "Bindings and job logs must stay outside training run directories")
    recipe = config["recipe"]
    require(type(recipe["batch_size"]) is int and recipe["batch_size"] > 0, "Invalid batch size")
    require(recipe.get("precision") in ("bf16", "fp32"), "Explicit bf16 or fp32 precision required")
    return config


def render(config_path, output, execute=False):
    config = validate_config(read_json(config_path), concrete=True)
    require(not output.exists(), "Job output directory exists; use a new version")
    scripts = {}
    for stage in STAGES:
        r = config["resources"]
        lines = ["#!/bin/bash -l", f"#$ -N mmaudio_{stage}", f"#$ -l h_rt={r['walltime']}",
                 f"#$ -l mem={r['mem_per_core']}", f"#$ -l tmpfs={r['tmpfs']}",
                 f"#$ -pe smp {r['cores']}", "#$ -cwd", "#$ -j y"]
        if stage != "verify":
            lines.append("#$ -l gpu=1")
        lines += ["set -euo pipefail", ': "${JOB_ID:?Submit this script with qsub after reviewing it}"',
                  ': "${NSLOTS:?Missing SGE CPU allocation}"', "export PYTHONNOUSERSITE=1 MPLBACKEND=Agg",
                  "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1",
                  'export OMP_NUM_THREADS="$NSLOTS"',
                  "export PYTHONPATH=" + shlex.quote(config["code_root"])]
        lines += ["module load " + shlex.quote(x) for x in config["modules"]]
        lines.append("cd " + shlex.quote(config["job_output"]))
        command = [config["python"], "-m", "fine_tune.myriad", "job", "--config", str(config_path.resolve()),
                   "--stage", stage, "--config-sha256", common.sha256_file(config_path), "--execute"]
        lines.append("exec " + shlex.join(command))
        scripts[f"{stage}.qsub"] = "\n".join(lines) + "\n"
    if execute:
        output.mkdir(parents=True, exist_ok=False)
        for name, content in scripts.items():
            (output / name).write_text(content)
    return dict(status="JOBS_RENDERED" if execute else "RENDER_PREVIEW", scripts=list(scripts), submitted=False)


def job(config_path, digest, stage, execute=False):
    require(common.sha256_file(config_path) == digest, "Job config changed since rendering")
    config = validate_config(read_json(config_path), concrete=True)
    if stage == "verify":
        command = [config["python"], "-m", "fine_tune.official_feature_bundle", "bind-training",
                   "--bundle", config["bundle_ready"], "--bundle-sha256", config["bundle_sha256"],
                   "--output", config["binding"], *weight_args(config)]
    else:
        command = training_command(config, stage)
        if stage == "resume":
            command += ["--resume", latest_checkpoint(config["train_run"])]
    if not execute:
        return dict(status="JOB_PREVIEW", command=shlex.join(command), submitted=False)
    require(os.environ.get("JOB_ID") and os.environ.get("SGE_O_WORKDIR"), "Heavy validation/training requires an SGE job")
    require(int(os.environ.get("NSLOTS", "0")) >= config["resources"]["cores"], "CPU allocation is too small")
    require(config.get("scheduler_verified") is True, "Confirm the actual Myriad scheduler first")
    require(Path(sys.executable).resolve() == Path(config["python"]).resolve(), "Use the reviewed Python environment")
    out = Path(config["job_output"]) / f"job_{os.environ['JOB_ID']}_{stage}"
    out.mkdir(parents=True, exist_ok=False)
    common.atomic_json(out / "STARTED.json", dict(stage=stage, config_sha256=digest,
                       command=command, job_id=os.environ["JOB_ID"], training_started=False))
    environment = dict(python=sys.version, executable=sys.executable,
                       modules=os.environ.get("LOADEDMODULES", ""), scheduler="SGE",
                       job_id=os.environ["JOB_ID"], slots=os.environ["NSLOTS"])
    for name, cmd in (("system", ["uname", "-a"]), ("packages", [sys.executable, "-m", "pip", "freeze"]),
                      ("gpu", ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"])):
        try:
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=30)
            environment[name] = dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
        except (OSError, subprocess.TimeoutExpired) as exc:
            environment[name] = dict(error=str(exc))
    common.atomic_json(out / "environment.json", environment)
    try:
        with (out / "job.log").open("x") as log:
            if stage in ("verify", "smoke"):
                subprocess.run([config["python"], "-m", "fine_tune.official_preflight",
                    "--official-repo", config["official_repo"], "--report", str(out / "preflight.json")],
                    check=True, stdout=log, stderr=subprocess.STDOUT)
            if stage == "smoke":
                subprocess.run(command + ["--stop-after", "20", "--execute"], check=True, stdout=log, stderr=subprocess.STDOUT)
                status = read_json(Path(config["smoke_run"]) / "training_status.json")
                require(status.get("phase") == "paused" and status.get("completed_updates") == 20,
                        "Checkpoint probe did not reach the controlled 20-step pause")
                checkpoint = latest_checkpoint(config["smoke_run"])
                subprocess.run(command + ["--resume", checkpoint, "--execute"], check=True, stdout=log, stderr=subprocess.STDOUT)
                smoke = read_json(Path(config["smoke_run"]) / "smoke_report.json")
                require(smoke.get("status") == "PASSED" and smoke.get("checkpoint_roundtrip_verified") is True,
                        "Checkpoint round-trip smoke did not pass")
            else:
                subprocess.run(command + ["--execute"], check=True, stdout=log, stderr=subprocess.STDOUT)
        # A paused training process is not a completed training run.
        phase = "verified" if stage == "verify" else read_json(Path(config["smoke_run"] if stage == "smoke" else config["train_run"]) / "training_status.json")["phase"]
        result = dict(status="STAGE_FINISHED", stage=stage, phase=phase, job_id=os.environ["JOB_ID"])
        common.atomic_json(out / "FINISHED.json", result)
        return result
    except BaseException as exc:
        common.atomic_json(out / "FAILED.json", dict(stage=stage, error=str(exc)))
        raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("render")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--execute", action="store_true", help="Write scripts only; never submit")
    p = sub.add_parser("job")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--config-sha256", required=True)
    p.add_argument("--stage", choices=STAGES, required=True)
    p.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    result = render(args.config, args.output, args.execute) if args.command == "render" else job(
        args.config, args.config_sha256, args.stage, args.execute)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
