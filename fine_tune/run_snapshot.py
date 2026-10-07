"""Hash-verified, append-only recovery snapshots for an ephemeral training VM.

Called synchronously while the local training writer lock is held. Unchanged
files reuse content-addressed objects. Only a completely verified snapshot can
replace LATEST.json; incomplete writes never become a recovery point.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
from uuid import uuid4

from fine_tune import official_common as common
from fine_tune.official_feature_bundle import member, read_json, require

SCHEMA = "mmaudio_run_snapshot_v1"
EXCLUDED = {".training.lock", "STOP_REQUESTED"}


def object_path(root, digest):
    require(isinstance(digest, str) and bool(re.fullmatch(r"[0-9a-f]{64}", digest)), "Invalid object SHA256")
    return member(root, "objects/" + digest)


def latest(root):
    root = Path(root).resolve()
    pointer = read_json(root / "LATEST.json")
    return member(root, pointer["ready"]), pointer["sha256"]


def verify(path, digest):
    path = Path(path).resolve()
    require(common.sha256_file(path) == digest, "Snapshot receipt SHA256 differs")
    require(path.name == "SNAPSHOT_READY.json" and path.parent.parent.name == "versions", "Invalid snapshot layout")
    root = path.parent.parent.parent
    receipt = read_json(path)
    require(receipt.get("schema") == SCHEMA and receipt.get("status") == "SNAPSHOT_VERIFIED", "Snapshot is incomplete")
    run = Path(receipt["run_dir"])
    require(run.is_absolute() and not run.is_relative_to(root), "Invalid original run path")
    files = receipt["files"]
    require(files and not EXCLUDED.intersection(files), "Snapshot includes ephemeral control files")
    for relative, entry in files.items():
        member(run, relative)
        obj = object_path(root, entry["sha256"])
        require(obj.stat().st_size == entry["bytes"] and common.sha256_file(obj) == entry["sha256"],
                "Snapshot object is missing or changed: " + relative)
    for required in ("run_metadata.json", "effective_config.json", "latest_checkpoint.json"):
        require(required in files, "Missing recovery metadata: " + required)
    require(any(k.startswith("ema_ckpts/") for k in files) or receipt.get("ema_directory_present") is True,
            "Missing EMA recovery directory")
    metadata = read_json(object_path(root, files["run_metadata.json"]["sha256"]))
    require(metadata["recipe"] == receipt["recipe"], "Snapshot recipe differs")
    checkpoint = read_json(object_path(root, files["latest_checkpoint.json"]["sha256"]))
    checkpoint_path = Path(checkpoint["path"])
    require(checkpoint_path.parent == run and checkpoint_path.name in files
            and files[checkpoint_path.name]["sha256"] == checkpoint["sha256"]
            and checkpoint["completed_updates"] == receipt["completed_updates"]
            and checkpoint["recipe"] == receipt["recipe"], "Snapshot checkpoint binding differs")
    sidecar = checkpoint_path.with_suffix(".json").name
    require(sidecar in files and read_json(object_path(root, files[sidecar]["sha256"])) == checkpoint,
            "Snapshot checkpoint sidecar differs")
    return receipt, root


def save(run, root):
    """The caller must hold the run's local writer lock; one VM per backup root."""
    run, root = Path(run).resolve(), Path(root).resolve()
    require(not root.is_relative_to(run) and not run.is_relative_to(root), "Backup and run directories must not overlap")
    metadata = read_json(run / "run_metadata.json")
    checkpoint = read_json(run / "latest_checkpoint.json")
    require(checkpoint["recipe"] == metadata["recipe"], "Checkpoint recipe differs from run")
    require((run / "ema_ckpts").is_dir(), "EMA directory is required")
    state = dict(schema=SCHEMA, run_dir=str(run), recipe=metadata["recipe"])
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "BACKUP_STATE.json"
    if state_path.exists():
        require(read_json(state_path) == state, "Backup root belongs to another run/recipe")
    else:
        require(not any(root.iterdir()), "Unknown files in backup root")
        common.atomic_json(state_path, state)
    if (root / "LATEST.json").exists():
        previous_path, previous_sha = latest(root)
        require(common.sha256_file(previous_path) == previous_sha, "Previous snapshot pointer changed")
        require(read_json(previous_path)["completed_updates"] <= checkpoint["completed_updates"],
                "Backup has a newer checkpoint; restore it or use a new run")
    inventory = {}
    for source in sorted(run.rglob("*")):
        require(not source.is_symlink(), "Symlinks are not accepted in a run snapshot")
        if not source.is_file() or source.name in EXCLUDED:
            continue
        require(not source.name.endswith((".pending", ".tmp")), "Partial run file retained; inspect before backup")
        digest = common.sha256_file(source)
        obj = object_path(root, digest)
        if obj.exists():
            require(common.sha256_file(obj) == digest, "Existing backup object is corrupt")
        else:
            obj.parent.mkdir(parents=True, exist_ok=True)
            temporary = obj.with_name(digest + ".pending-" + uuid4().hex)
            shutil.copyfile(source, temporary)
            require(common.sha256_file(temporary) == digest, "Backup copy readback differs")
            temporary.replace(obj)
        inventory[source.relative_to(run).as_posix()] = dict(sha256=digest, bytes=source.stat().st_size)
    receipt = dict(schema=SCHEMA, status="SNAPSHOT_VERIFIED", run_dir=str(run), recipe=metadata["recipe"],
                   completed_updates=checkpoint["completed_updates"], files=inventory, ema_directory_present=True)
    path = root / "versions" / f"step_{checkpoint['completed_updates']:09d}_{uuid4().hex}" / "SNAPSHOT_READY.json"
    common.atomic_json(path, receipt)
    digest = common.sha256_file(path)
    verify(path, digest)
    common.atomic_json(root / "LATEST.json", dict(ready=path.relative_to(root).as_posix(), sha256=digest,
                       completed_updates=checkpoint["completed_updates"]))
    return dict(ready=str(path), sha256=digest, completed_updates=checkpoint["completed_updates"])


def restore(path, digest, output, execute=False):
    receipt, root = verify(path, digest)
    output = Path(output).resolve()
    require(output == Path(receipt["run_dir"]), "Restore to the same absolute run path to preserve receipt identity")
    require(not output.exists(), "Restore destination exists; inspect it, never overwrite")
    if execute:
        output.mkdir(parents=True, exist_ok=False)
        for relative, entry in receipt["files"].items():
            target = member(output, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(object_path(root, entry["sha256"]), target)
            require(common.sha256_file(target) == entry["sha256"], "Restored file readback differs")
        (output / "ema_ckpts").mkdir(exist_ok=True)
    return dict(status="RUN_RESTORED" if execute else "RESTORE_PREVIEW", run_dir=str(output),
                completed_updates=receipt["completed_updates"], training_started=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    path, digest = latest(args.backup_root)
    print(restore(path, digest, args.output, args.execute))


if __name__ == "__main__":
    main()
