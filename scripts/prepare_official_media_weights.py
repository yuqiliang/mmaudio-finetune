"""Explicit, hash-checked local encoder preparation; no encoder forward/GPU.

Defaults to dry-run. Use only in an authorised clean Colab workspace. Existing
files are reused only after validation; mismatches are never overwritten.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fine_tune import official_common as common
from fine_tune import official_extract as shared

CLIP_REVISION = "01b771ed0d1395ca5ffdd279897d665ebe00dfd2"
VOCODER_REVISION = "95a9d1dcb12906c03edd938d77b9333d6ded7dfb"


def verify_existing_report(args, official):
    """Recheck a published set without network calls or replacing any bytes."""
    report = json.loads(args.report.read_text())
    if (report.get("status") != "ENCODER_FILES_VERIFIED_NOT_EXTRACTED"
            or report.get("official_sources") != official
            or report.get("training_ready") is not False
            or report.get("gpu_compute_started") is not False):
        raise ValueError("Existing weight report has a different scope/source identity")
    saved = report["weights"]
    for key, revision in (("clip", CLIP_REVISION), ("vocoder44k", VOCODER_REVISION)):
        if saved[key]["revision"] != revision:
            raise ValueError("Existing weight report uses another pinned HF revision")
    args.clip_snapshot = Path(saved["clip"]["snapshot"])
    args.clip_revision = CLIP_REVISION
    args.vocoder_snapshot = Path(saved["vocoder44k"]["snapshot"])
    args.vocoder_revision = VOCODER_REVISION
    current = shared.inventory_weights(args, common.MODEL_SPECS["small_44k"])
    if current != saved:
        raise ValueError("Existing encoder bytes/locations changed; no files overwritten or downloaded")
    print("Existing encoder report and files verified; no downloads:", args.report)


def prepare(args):
    official = common.verify_official_repo(args.official_repo)
    tree = ast.parse((args.official_repo / "mmaudio/utils/download_utils.py").read_text())
    links = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "links" for t in node.targets))
    catalog = {row["name"]: row for row in links}
    print(json.dumps(dict(mode="prepare_weights" if args.execute else "dry_run", gpu_compute=False,
        weights_dir=str(args.weights_dir), cache_dir=str(args.cache_dir),
        clip_revision=CLIP_REVISION, vocoder_revision=VOCODER_REVISION), indent=2))
    if not args.execute:
        return
    if sys.platform != "linux" or not all(p.resolve().is_relative_to("/content")
            for p in (args.weights_dir, args.cache_dir, args.report)):
        raise ValueError("This explicit setup writes only inside a Colab /content workspace")
    if args.report.exists():
        verify_existing_report(args, official)
        return
    args.weights_dir.mkdir(parents=True, exist_ok=True)
    for name in ("v1-44.pth", "synchformer_state_dict.pth"):
        target = args.weights_dir / name
        if not target.exists():
            temporary = target.with_suffix(".pth.downloading")
            if temporary.exists():
                raise ValueError(f"Partial download retained; inspect before retry: {temporary}")
            source = args.source_weights_dir / name if args.source_weights_dir else None
            with temporary.open("xb") as output:
                if source and source.is_file():
                    with source.open("rb") as input_file:
                        shutil.copyfileobj(input_file, output)
                else:
                    with urllib.request.urlopen(catalog[name]["url"], timeout=120) as response:
                        shutil.copyfileobj(response, output)
            with temporary.open("rb") as stream:
                if hashlib.file_digest(stream, "md5").hexdigest() != catalog[name]["md5"]:
                    raise ValueError(f"Downloaded weight MD5 mismatch; retained: {temporary}")
            # Publish without POSIX rename's overwrite behaviour if another
            # writer created this name while the download was in flight.
            os.link(temporary, target)
            temporary.unlink()
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "md5").hexdigest() != catalog[name]["md5"]:
                raise ValueError(f"Existing weight mismatch; not overwriting: {target}")
    from huggingface_hub import snapshot_download
    args.clip_snapshot = Path(snapshot_download(shared.CLIP_REPO, revision=CLIP_REVISION,
        cache_dir=args.cache_dir, allow_patterns=["open_clip_config.json", "open_clip_pytorch_model.bin"]))
    args.clip_revision = CLIP_REVISION
    args.vocoder_snapshot = Path(snapshot_download(shared.VOCODER_REPO, revision=VOCODER_REVISION,
        cache_dir=args.cache_dir, allow_patterns=["config.json", "bigvgan_generator.pt"]))
    args.vocoder_revision = VOCODER_REVISION
    weights = shared.inventory_weights(args, common.MODEL_SPECS["small_44k"])
    if common.verify_official_repo(args.official_repo) != official:
        raise ValueError("Official source changed during encoder preparation")
    report = dict(status="ENCODER_FILES_VERIFIED_NOT_EXTRACTED", official_sources=official,
        weights=weights, training_ready=False, gpu_compute_started=False)
    if args.report.exists():
        if json.loads(args.report.read_text()) != report:
            raise ValueError("Weight report changed; choose a new explicit version")
    else:
        common.atomic_json(args.report, report)
    print("Verified encoder files; no GPU computation:", args.report)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--official-repo", type=Path, required=True)
    p.add_argument("--weights-dir", type=Path, required=True)
    p.add_argument("--source-weights-dir", type=Path)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    return p


def main(argv=None):
    prepare(parser().parse_args(argv))


if __name__ == "__main__":
    main()
