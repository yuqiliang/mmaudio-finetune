"""Platform-independent arguments for the guarded official training launcher."""
from pathlib import Path

from fine_tune import official_common as common
from fine_tune.official_feature_bundle import read_json, require


def weight_args(config):
    return [value for key in ("official_repo", "weights_dir", "clip_snapshot", "clip_revision", "vocoder_snapshot", "vocoder_revision")
            for value in ("--" + key.replace("_", "-"), config[key])]


def training_command(config, stage):
    r = config["recipe"]
    smoke = stage == "smoke"
    if not smoke:
        require(type(r.get("steps")) is int and r["steps"] > 0, "Confirm formal target updates before training")
    command = [config["python"], "-m", "fine_tune.official_train", "--extraction-ready", config["binding"],
               "--official-repo", config["official_repo"], "--weights", config["pretrained"],
               "--run-dir", config["smoke_run"] if smoke else config["train_run"],
               "--steps", "24" if smoke else str(r["steps"])]
    for key in ("batch_size", "learning_rate", "warmup_steps", "seed", "weight_decay", "gradient_clip", "num_workers"):
        command += ["--" + key.replace("_", "-"), str(r[key])]
    command += ["--save-every", "20" if smoke else str(r["save_every"]),
                "--val-every", "24" if smoke else str(r["val_every"]),
                "--ema-every", "24" if smoke else str(r["ema_every"])]
    if "val_batch_size" in r:
        command += ["--val-batch-size", str(r["val_batch_size"])]
    if r["precision"] == "fp32":
        command.append("--fp32")
    if smoke:
        command += ["--smoke", "--checkpoint-probe"]
    else:
        command += ["--smoke-report", str(Path(config["smoke_run"]) / "smoke_report.json")]
    return command


def latest_checkpoint(run):
    run = Path(run).resolve()
    value = read_json(run / "latest_checkpoint.json")
    checkpoint = Path(value["path"]).resolve()
    require(checkpoint.parent == run and common.sha256_file(checkpoint) == value["sha256"],
            "Checkpoint path/hash differs")
    return str(checkpoint)
