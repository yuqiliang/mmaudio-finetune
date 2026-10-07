"""Read-only environment checks before official extraction or training.

No model is constructed, no weights are downloaded and no CUDA kernels are run.
This checks imports and FFmpeg shared-library availability, not data quality or
numerical correctness. --report optionally saves the diagnostic JSON.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fine_tune.official_common import activate_official_repo, atomic_json, sha256_file


def diagnose(repo: Path) -> dict:
    report = {"utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
              "python_executable": sys.executable, "checks": [], "packages": {},
              "model_initialized": False, "gpu_compute_started": False, "download_started": False}

    def check(name, action):
        try:
            result = action()
            report["checks"].append({"name": name, "passed": True, "detail": result})
            return result
        except Exception as exc:
            report["checks"].append({"name": name, "passed": False,
                                      "error": f"{type(exc).__name__}: {exc}"})
            return None

    provenance = check("pinned_clean_official_repository", lambda: activate_official_repo(repo))
    if provenance is None:
        report["status"] = "BLOCKED"
        return report
    # Import-time lookups must remain offline; no replacing a missing av-bench.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    for package in ("torch", "torchvision", "torchaudio", "tensordict", "open_clip_torch",
                    "hydra-core", "nitrous-ema", "numpy", "huggingface-hub", "av_bench"):
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None

    def required_symbol(module_name, symbol=None):
        module = importlib.import_module(module_name)
        if symbol and not hasattr(module, symbol):
            raise RuntimeError(f"Required symbol {module_name}.{symbol} is unavailable")
        return {"module": module_name, "path": getattr(module, "__file__", None), "symbol": symbol}

    for module, symbol in (("torch", None), ("torchvision", None), ("torchaudio", None),
                           ("torio.io", "StreamingMediaDecoder"), ("tensordict", "TensorDict"),
                           ("hydra", "compose"), ("nitrous_ema", "PostHocEMA"),
                           ("open_clip", "create_model_from_pretrained"),
                           ("av_bench.extract", "extract"), ("av_bench.evaluate", "evaluate"),
                           ("mmaudio.data.extraction.vgg_sound", "VGGSound"),
                           ("mmaudio.data.extracted_vgg", "ExtractedVGG"),
                           ("mmaudio.runner", "Runner")):
        check(f"import_{module}", lambda module=module, symbol=symbol: required_symbol(module, symbol))

    def torch_audio_pair():
        versions = report["packages"]
        torch_version, audio_version = versions["torch"], versions["torchaudio"]
        if not torch_version or not audio_version or torch_version.split("+")[0] != audio_version.split("+")[0]:
            raise RuntimeError("torch and torchaudio must have matching release versions")
        return {"torch": torch_version, "torchaudio": audio_version}
    check("torch_torchaudio_version_pair", torch_audio_pair)

    def ffmpeg_binary():
        executable = shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError("ffmpeg is not on PATH; the official decoder requires FFmpeg < 7")
        result = subprocess.run([executable, "-version"], capture_output=True, text=True,
                                timeout=15, check=True)
        first_line = result.stdout.splitlines()[0]
        version = re.search(r"ffmpeg version (?:n)?(\d+)", first_line)
        if not version or int(version.group(1)) >= 7:
            raise RuntimeError(f"Expected ffmpeg < 7 for the pinned torio pipeline; found: {first_line}")
        return {"path": executable, "version": first_line}
    check("ffmpeg_binary", ffmpeg_binary)

    def ffmpeg_shared():
        utils = importlib.import_module("torio.utils.ffmpeg_utils")
        versions = utils.get_versions()  # load torio's extension, not just CLI ffmpeg
        if not versions or int(versions.get("libavcodec", [999])[0]) >= 61:
            raise RuntimeError(f"Unsupported FFmpeg libraries for official torio: {versions}")
        video = utils.get_video_decoders()
        audio = utils.get_audio_decoders()
        if "h264" not in video or "aac" not in audio or "pcm_s16le" not in audio:
            raise RuntimeError("FFmpeg shared decoder lacks H.264, AAC or PCM support")
        return {"versions": versions, "h264": True, "aac": True, "pcm_s16le": True}
    check("torio_shared_ffmpeg_decoder", ffmpeg_shared)

    def empty_string():
        path = repo / "ext_weights/empty_string.pth"
        if not path.is_file():
            raise RuntimeError("Official empty_string.pth is missing in ext_weights; training cannot initialize")
        return {"path": str(path), "sha256": sha256_file(path)}
    check("empty_string_weights", empty_string)

    # Driver availability is informative; no tensors/models are placed on CUDA.
    def cuda_information():
        torch = importlib.import_module("torch")
        return {"torch_cuda_build": torch.version.cuda, "available": torch.cuda.is_available()}
    report["cuda"] = check("cuda_driver_information", cuda_information)
    passed = all(entry["passed"] for entry in report["checks"])
    report["status"] = "ENVIRONMENT_CHECKS_PASSED" if passed else "BLOCKED"
    report["next_step"] = ("Run official extraction dry-run after reviewed clips and pinned weights are ready. "
                           "GPU feature extraction and the 20–100-update smoke remain untested by this diagnostic.")
    report["compatibility_candidate"] = ("Python 3.10/3.11, torch 2.5.1, torchvision 0.20.1, "
                                          "torchaudio 2.5.1, FFmpeg 6.x, open_clip_torch 2.29.0. "
                                          "This is a setup candidate, not an environment lock proven on your new data.")
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official-repo", required=True, type=Path)
    parser.add_argument("--report", type=Path, help="Optional diagnostic JSON destination")
    args = parser.parse_args(argv)
    result = diagnose(args.official_repo.resolve())
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.report:
        atomic_json(args.report, result)
    return 0 if result["status"] == "ENVIRONMENT_CHECKS_PASSED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
