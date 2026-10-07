"""CPU-only guards for official extraction; no torch, downloads, or GPU required."""
from __future__ import annotations

import csv
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from fine_tune import official_extract as extraction


def hash_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class FakeTensor:
    """Minimal shape/range object: tests inspect guards, not numerical kernels."""
    def __init__(self, shape, low=-0.5, high=0.5, finite=True, nonzero=True, floating=True):
        self.shape, self.low, self.high = shape, low, high
        self.finite, self.nonzero, self.floating = finite, nonzero, floating
        self.dtype = "torch.float32"

    def is_floating_point(self): return self.floating
    def isfinite(self): return SimpleNamespace(all=lambda: self.finite)
    def min(self): return self.low
    def max(self): return self.high
    def ne(self, number): return SimpleNamespace(any=lambda: self.nonzero)
    def lt(self, number): return SimpleNamespace(any=lambda: self.low < number)


SPEC = {"latent_seq_len": 250, "latent_dim": 20, "audio_samples": 128000}


class ExtractionGuardsTests(unittest.TestCase):
    def test_help_does_not_require_gpu_libraries(self):
        with self.assertRaises(SystemExit) as result:
            extraction.parser().parse_args(["--help"])
        self.assertEqual(result.exception.code, 0)

    def test_model_shapes_are_not_shared_between_16k_and_44k(self):
        self.assertEqual(extraction.feature_shapes(SPEC)["mean"], (250, 20))
        self.assertEqual(extraction.feature_shapes({"latent_seq_len": 345, "latent_dim": 40})["mean"], (345, 40))
        self.assertEqual(extraction.feature_shapes(SPEC)["sync_features"], (192, 768))

    def test_sample_shape_and_range_allow_bicubic_overshoot(self):
        sample = {"id": "clip", "caption": "urban soundscape",
                  "audio": FakeTensor((128000,)),
                  "clip_video": FakeTensor((64, 3, 384, 384), -0.1, 1.1),
                  "sync_video": FakeTensor((200, 3, 224, 224), -1.2, 1.2)}
        row = {"clip_id": "clip", "label": "urban soundscape"}
        report = extraction.validate_sample(sample, row, SPEC)
        self.assertEqual(report["sync_video"]["shape"][0], 200)
        sample["sync_video"] = FakeTensor((200, 3, 224, 224), 0, 255)
        with self.assertRaisesRegex(ValueError, "bounds"):
            extraction.validate_sample(sample, row, SPEC)

    def test_sample_rejects_nan_and_wrong_pairing(self):
        row = {"clip_id": "clip", "label": "label"}
        with self.assertRaisesRegex(ValueError, "ID/caption"):
            extraction.validate_sample({"id": "another", "caption": "label"}, row, SPEC)
        sample = {"id": "clip", "caption": "label", "audio": FakeTensor((128000,), finite=False)}
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            extraction.validate_sample(sample, row, SPEC)

    def test_features_reject_negative_std_and_wrong_modalities(self):
        values = {name: FakeTensor(shape, 0.1, 0.5) for name, shape in extraction.feature_shapes(SPEC).items()}
        extraction.validate_features(values, SPEC)
        values["std"] = FakeTensor((250, 20), -0.1, 0.5)
        with self.assertRaisesRegex(ValueError, "negative"):
            extraction.validate_features(values, SPEC)
        values.pop("sync_features")
        with self.assertRaisesRegex(ValueError, "five"):
            extraction.validate_features(values, SPEC)

    def test_receipt_binds_id_order_source_and_plan(self):
        row = {"clip_id": "clip", "label": "urban", "sha256": "source"}
        receipt = {"run_id": "run", "row_index": 0, "clip_id": "clip", "label": "urban",
                   "source_sha256": "source", "status": "COMPLETE"}
        self.assertTrue(extraction.row_receipt_valid(receipt, row, 0, "run"))
        self.assertFalse(extraction.row_receipt_valid(receipt, row, 1, "run"))
        self.assertFalse(extraction.row_receipt_valid(receipt, row, 0, "other"))
        self.assertFalse(extraction.row_receipt_valid(receipt, {**row, "sha256": "changed"}, 0, "run"))

    def test_output_lock_refuses_concurrent_writer_and_releases(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with extraction.exclusive_output(output):
                with self.assertRaisesRegex(RuntimeError, "Another extraction"):
                    with extraction.exclusive_output(output):
                        pass
            with extraction.exclusive_output(output):
                self.assertTrue((output / ".extraction.lock").is_file())

    def test_official_weight_catalog_reads_literals_without_running_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            repo = Path(temporary)
            module = repo / "mmaudio/utils/download_utils.py"
            module.parent.mkdir(parents=True)
            module.write_text("raise RuntimeError('must never execute')\nlinks = ["
                              "{'name': 'v1-16.pth', 'md5': '" + "a" * 32 + "'}]\n")
            self.assertEqual(extraction.official_weight_catalog(repo), {"v1-16.pth": "a" * 32})

    def test_wrong_official_encoder_weight_is_refused_before_model_load(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "v1-16.pth").write_bytes(b"wrong checkpoint")
            args = SimpleNamespace(weights_dir=root, official_repo=root)
            spec = {"vae_filename": "v1-16.pth", "vocoder_filename": None, "mode": "16k"}
            with patch.object(extraction, "official_weight_catalog", return_value={"v1-16.pth": "a" * 32}):
                with self.assertRaisesRegex(ValueError, "differs from published"):
                    extraction.inventory_weights(args, spec)

    def test_probe_never_publishes_training_ready_and_can_resume_full(self):
        from fine_tune.official_common import atomic_json
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            output = base / "output"
            output.mkdir()
            plan_path = base / "plan.json"
            plan_path.write_text("{}")
            plan = {"splits": {split: {} for split in ("train", "val", "test")},
                    "dataset_version": "test", "model": "small_16k", "preprocessing_id": "official-test",
                    "split_policy": "source", "normalize_audio": {"train": True, "val": False, "test": False}}
            rows = {split: [{"clip_id": split + str(i)} for i in range(3)] for split in plan["splits"]}
            weights, official = {"vae": {"path": "vae"}, "synchformer": {"path": "sync"}}, {"commit": "test"}
            identity = {"plan_sha256": hash_file(plan_path)}
            common = SimpleNamespace(sha256_file=hash_file, atomic_json=atomic_json, PINNED_COMMIT="test",
                load_plan=lambda _: plan, verify_official_repo=lambda _: official)
            args = SimpleNamespace(resume=False, probe_per_split=1, official_repo=base,
                                   clip_batch_size=8, sync_batch_size=1)
            torch = SimpleNamespace(backends=SimpleNamespace(
                cuda=SimpleNamespace(matmul=SimpleNamespace()), cudnn=SimpleNamespace()))
            options = dict(plan_path=plan_path, plan=plan, spec={"mode": "16k"}, official=official,
                           weights=weights, rows=rows, output=output, versions={}, identity=identity,
                           run_id="identical", torch=torch, dataset_class=MagicMock(), features_class=MagicMock())
            with patch.object(extraction, "_common", return_value=common), \
                 patch.object(extraction, "local_encoder_resolution", return_value=nullcontext()), \
                 patch.object(extraction, "extract_split") as extract, \
                 patch.object(extraction, "build_memmap", return_value={"count": 3}) as build, \
                 patch.object(extraction, "validate_manifest_sources", return_value=rows), \
                 patch.object(extraction, "inventory_weights", return_value=weights):
                self.assertEqual(extraction.run_extraction(args, **options), 0)
                build.assert_not_called()
                self.assertEqual([call.kwargs["limit"] for call in extract.call_args_list], [1, 1, 1])
                probe = json.loads((output / "PROBE_COMPLETE.json").read_text())
                self.assertFalse(probe["training_ready"])
                self.assertFalse((output / "extraction_READY.json").exists())
                args.resume, args.probe_per_split = True, None
                self.assertEqual(extraction.run_extraction(args, **options), 0)
                self.assertEqual(build.call_count, 3)
                ready = json.loads((output / "extraction_READY.json").read_text())
                self.assertEqual(ready["status"], "READY_FOR_TRAINING")
                self.assertEqual(ready["run_id"], probe["run_id"])

    def test_local_snapshot_requires_exact_revision_and_expected_architecture(self):
        revision = "a" * 40
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / revision
            snapshot.mkdir()
            (snapshot / "open_clip_pytorch_model.bin").write_bytes(b"local test weight")
            # Real pinned apple/*-384 config, not a guessed dimension from its name.
            config = {"model_cfg": {"embed_dim": 1024, "quick_gelu": True,
                      "vision_cfg": {"image_size": 378, "layers": 32, "width": 1280,
                                     "head_width": 80, "patch_size": 14},
                      "text_cfg": {"context_length": 77, "vocab_size": 49408,
                                   "width": 1024, "heads": 16, "layers": 24}},
                      "preprocess_cfg": {"mean": [0.48145466, 0.4578275, 0.40821073],
                          "std": [0.26862954, 0.26130258, 0.27577711],
                          "interpolation": "bicubic", "resize_mode": "squash"}}
            (snapshot / "open_clip_config.json").write_text(json.dumps(config))
            with patch.object(extraction, "_common", return_value=SimpleNamespace(sha256_file=hash_file)):
                result = extraction.snapshot_inventory(snapshot, revision, kind="clip")
                self.assertIn("open_clip_pytorch_model.bin", result["files"])
                with self.assertRaisesRegex(ValueError, "40-character"):
                    extraction.snapshot_inventory(snapshot, "main", kind="clip")
                config["model_cfg"]["vision_cfg"]["image_size"] = 384
                (snapshot / "open_clip_config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "architecture"):
                    extraction.snapshot_inventory(snapshot, revision, kind="clip")
                config["model_cfg"]["vision_cfg"]["image_size"] = 378
                config["preprocess_cfg"]["resize_mode"] = "shortest"
                (snapshot / "open_clip_config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "preprocess_cfg"):
                    extraction.snapshot_inventory(snapshot, revision, kind="clip")
                config["preprocess_cfg"]["resize_mode"] = "squash"
                config["model_cfg"]["embed_dim"] = 512
                (snapshot / "open_clip_config.json").write_text(json.dumps(config))
                with self.assertRaisesRegex(ValueError, "architecture"):
                    extraction.snapshot_inventory(snapshot, revision, kind="clip")

    def test_source_preflight_rejects_modified_and_unreviewed_clips(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            plan = {"splits": {}}
            for split in ("train", "val", "test"):
                video = root / (split + ".mp4")
                video.write_bytes(split.encode())
                tsv = root / (split + ".tsv")
                tsv.write_text(f"id\tlabel\n{split}\turban\n")
                manifest = root / (split + ".csv")
                row = {"clip_id": split, "label": "urban", "path": str(video), "sha256": hash_file(video),
                       "split": split, "manual_review_status": "included", "qc_status": "pass"}
                with manifest.open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerow(row)
                plan["splits"][split] = {"manifest": str(manifest), "tsv": str(tsv), "count": 1, "video_root": str(root)}
            with patch.object(extraction, "_common", return_value=SimpleNamespace(sha256_file=hash_file)):
                self.assertEqual(len(extraction.validate_manifest_sources(plan)), 3)
                (root / "train.mp4").write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "bytes changed"):
                    extraction.validate_manifest_sources(plan)

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("tensordict"),
                         "Optional real memmap round-trip needs torch+tensordict; no GPU required")
    def test_real_memmap_streaming_roundtrip_and_published_tamper(self):
        import torch
        import tensordict as td
        from fine_tune.official_common import atomic_json
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            cache = output / "row_cache" / "train"
            cache.mkdir(parents=True)
            rows = []
            for index in range(2):
                row = {"clip_id": f"clip{index}", "label": "urban", "sha256": f"source{index}"}
                rows.append(row)
                values = {key: torch.ones(shape, dtype=torch.float32) * (index + 1)
                          for key, shape in extraction.feature_shapes(SPEC).items()}
                data_path, receipt_path = extraction.row_paths(cache, index)
                torch.save(values, data_path)
                atomic_json(receipt_path, {"status": "COMPLETE", "run_id": "test", "row_index": index,
                    "clip_id": row["clip_id"], "label": "urban", "source_sha256": row["sha256"],
                    "feature_sha256": hash_file(data_path)})
            tsv, manifest = output / "input.tsv", output / "input.csv"
            tsv.write_text("id\tlabel\nclip0\turban\nclip1\turban\n")
            manifest.write_text("clip_id,label\nclip0,urban\nclip1,urban\n")
            result = extraction.build_memmap("train", rows, {"tsv": str(tsv), "manifest": str(manifest)},
                    spec=SPEC, output=output, run_id="test", torch=torch)
            self.assertEqual(result["count"], 2)
            self.assertIn("meta.json", result["feature_checksums"])
            # Completed maps may be re-verified, but must never silently change.
            extraction.build_memmap("train", rows, {"tsv": str(tsv), "manifest": str(manifest)},
                    spec=SPEC, output=output, run_id="test", torch=torch)
            changed = td.TensorDict.load_memmap(result["memmap_dir"])
            changed["mean"][0, 0, 0] = 999
            del changed
            with self.assertRaisesRegex(ValueError, "Published memmap differs"):
                extraction.build_memmap("train", rows, {"tsv": str(tsv), "manifest": str(manifest)},
                        spec=SPEC, output=output, run_id="test", torch=torch)


if __name__ == "__main__":
    unittest.main()
