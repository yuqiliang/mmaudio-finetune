"""CPU-only guard tests; no model packages, downloads, or GPU use."""
from __future__ import annotations

import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fine_tune import official_train as training


class TrainingGuards(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "official"
        (self.repo / "mmaudio/utils").mkdir(parents=True)
        (self.repo / "ext_weights").mkdir()
        self.weights = self.repo / "weights.pth"
        self.weights.write_bytes(b"official-test-fixture")
        expected = hashlib.md5(self.weights.read_bytes()).hexdigest()
        (self.repo / "mmaudio/utils/download_utils.py").write_text(
            f"links = [{{'name': 'mmaudio_small_16k.pth', 'md5': '{expected}'}}]\n")
        (self.repo / "ext_weights/empty_string.pth").write_bytes(b"empty-string")
        source = self.repo / "mmaudio/utils/download_utils.py"
        self.provenance = {"commit": training.PINNED_COMMIT,
                           "files": {"mmaudio/utils/download_utils.py": training.sha256_file(source)}}
        self.plan = {"model": "small_16k", "preprocessing_id": "official-test"}
        plan_file = self.root / "plan.json"
        plan_file.write_text(json.dumps(self.plan))
        encoders = {}
        for name in ("vae", "synchformer", "vocoder16k", "clip"):
            member = self.root / f"{name}.pth"
            member.write_bytes(name.encode())
            encoders[name] = {"path": str(member), "sha256": training.sha256_file(member)}
        self.ready = {"schema_version": 1, "status": "READY_FOR_TRAINING",
                      "official_commit": training.PINNED_COMMIT,
                      "model": "small_16k", "preprocessing_id": "official-test",
                      "plan_path": str(plan_file), "plan_sha256": training.sha256_file(plan_file),
                      "official_sources": self.provenance, "encoder_hashes": encoders, "splits": {}}
        for split in ("train", "val", "test"):
            memmap = self.root / split
            memmap.mkdir()
            blob = memmap / "mean.memmap"
            blob.write_bytes(f"features-{split}".encode())
            tsv = self.root / f"{split}.tsv"
            manifest = self.root / f"{split}.csv"
            with tsv.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["id", "label"], delimiter="\t")
                writer.writeheader()
                writer.writerows({"id": f"{split}{i}", "label": "urban soundscape"} for i in range(2))
            with manifest.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["clip_id", "label", "split"])
                writer.writeheader()
                writer.writerows({"clip_id": f"{split}{i}", "label": "urban soundscape", "split": split} for i in range(2))
            self.ready["splits"][split] = {
                "count": 2, "tsv": str(tsv), "tsv_sha256": training.sha256_file(tsv),
                "manifest": str(manifest), "manifest_sha256": training.sha256_file(manifest),
                "memmap_dir": str(memmap), "feature_checksums": {"mean.memmap": training.sha256_file(blob)}}
        self.ready_file = self.root / "extraction_READY.json"
        self.plan["splits"] = {name: dict(entry) for name, entry in self.ready["splits"].items()}
        self.write_ready()
        self.patches = [patch.object(training, "verify_official_repo", return_value=self.provenance),
                        patch.object(training, "load_plan", return_value=self.plan)]
        for mocked in self.patches:
            mocked.start()
            self.addCleanup(mocked.stop)

    def write_ready(self):
        self.ready_file.write_text(json.dumps(self.ready))

    def args(self, *extra):
        return training.parser().parse_args([
            "--extraction-ready", str(self.ready_file), "--official-repo", str(self.repo),
            "--weights", str(self.weights), "--run-dir", str(self.root / "new-run"),
            "--steps", "20", "--smoke", *extra])

    def test_valid_cpu_preflight_does_not_create_run_or_import_torch(self):
        import sys
        before = "torch" in sys.modules
        result = training.preflight(self.args())
        self.assertEqual(result["ready"]["splits"]["train"]["count"], 2)
        self.assertFalse((self.root / "new-run").exists())
        self.assertEqual(before, "torch" in sys.modules)

    def test_legacy_custom_preprocessing_rejected(self):
        self.ready["preprocessing_id"] = "custom-old"
        self.write_ready()
        with self.assertRaisesRegex(ValueError, "Preprocessing"):
            training.preflight(self.args())

    def test_training_and_extraction_code_must_match(self):
        self.ready["official_sources"]["files"] = {}
        self.write_ready()
        with self.assertRaisesRegex(ValueError, "provenance"):
            training.preflight(self.args())

    def test_finetuned_initialization_rejected(self):
        self.weights.write_bytes(b"old-pilot-checkpoint")
        with self.assertRaisesRegex(ValueError, "official pretrained"):
            training.preflight(self.args())

    def test_feature_mutation_rejected(self):
        (self.root / "train/mean.memmap").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            training.preflight(self.args())

    def test_unrecorded_feature_file_rejected(self):
        (self.root / "train/unrecorded.memmap").write_bytes(b"unrecorded")
        with self.assertRaisesRegex(ValueError, "every memmap file"):
            training.preflight(self.args())

    def test_tsv_and_manifest_row_order_enforced_even_with_new_hash(self):
        entry = self.ready["splits"]["train"]
        tsv = Path(entry["tsv"])
        tsv.write_text("id\tlabel\ntrain1\turban soundscape\ntrain0\turban soundscape\n")
        entry["tsv_sha256"] = training.sha256_file(tsv)
        self.write_ready()
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            training.preflight(self.args())

    def test_manifest_cannot_silently_switch_from_frozen_plan(self):
        self.ready["splits"]["train"]["count"] = 1
        self.write_ready()
        with self.assertRaisesRegex(ValueError, "frozen plan"):
            training.preflight(self.args())

    def test_shared_split_memmap_rejected(self):
        self.ready["splits"]["val"]["memmap_dir"] = self.ready["splits"]["train"]["memmap_dir"]
        self.write_ready()
        with self.assertRaisesRegex(ValueError, "separate"):
            training.preflight(self.args())

    def test_existing_run_never_implicitly_resumed(self):
        (self.root / "new-run").mkdir()
        with self.assertRaisesRegex(ValueError, "already exists"):
            training.preflight(self.args())

    def test_smoke_step_bounds(self):
        with self.assertRaisesRegex(ValueError, "20–100"):
            training.preflight(self.args("--steps", "1"))

    def test_formal_requires_matching_smoke(self):
        args = self.args("--steps", "500")
        args.smoke = False
        with self.assertRaisesRegex(ValueError, "smoke-report"):
            training.preflight(args)

    def test_ema_final_step_has_history(self):
        args = self.args("--steps", "501")
        args.smoke = False
        with self.assertRaisesRegex(ValueError, "divisible"):
            training.preflight(args)

    def test_smoke_receipt_must_match_recipe(self):
        accepted = self.root / "smoke_report.json"
        accepted.write_text(json.dumps({"status": "PASSED", "recipe": {"old": True}}))
        with self.assertRaisesRegex(ValueError, "another"):
            training.verify_smoke(accepted, {"new": True})

    def test_default_main_never_calls_execution(self):
        argv = ["--extraction-ready", str(self.ready_file), "--official-repo", str(self.repo),
                "--weights", str(self.weights), "--run-dir", str(self.root / "new-run"),
                "--steps", "20", "--smoke"]
        with patch.object(training, "execute") as execute:
            self.assertEqual(training.main(argv), 0)
            execute.assert_not_called()

    def test_concurrent_resume_cannot_enter_training(self):
        import fcntl
        args = self.args()
        checked = training.preflight(args)
        run_dir = Path(checked["run_dir"])
        run_dir.mkdir()
        args.resume = "explicit-resume-fixture"
        with (run_dir / ".training.lock").open("a+") as owner:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with patch.object(training, "_execute_locked") as execute:
                with self.assertRaisesRegex(RuntimeError, "already owns"):
                    training.execute(args, checked)
                execute.assert_not_called()

    def test_environment_diagnostic_stops_before_imports_for_wrong_repo(self):
        from fine_tune import official_preflight
        with patch.object(official_preflight, "activate_official_repo", side_effect=ValueError("wrong commit")):
            with patch.object(official_preflight.importlib, "import_module") as importer:
                result = official_preflight.diagnose(self.repo)
                self.assertEqual(result["status"], "BLOCKED")
                self.assertFalse(result["gpu_compute_started"])
                importer.assert_not_called()

    def test_failed_resume_initialization_replaces_stale_status(self):
        args = self.args()
        checked = training.preflight(args)
        run_dir = Path(checked["run_dir"])
        run_dir.mkdir()
        status = run_dir / "training_status.json"
        status.write_text(json.dumps({"phase": "paused", "completed_updates": 42}))
        args.resume = "fixture"
        with patch.object(training, "_execute_locked", side_effect=RuntimeError("Missing dependency")):
            with self.assertRaisesRegex(RuntimeError, "Missing dependency"):
                training.execute(args, checked)
        result = json.loads(status.read_text())
        self.assertEqual(result["phase"], "initialization_failed")
        self.assertEqual(result["completed_updates"], 42)


if __name__ == "__main__":
    unittest.main()
