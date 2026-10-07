import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fine_tune import colab_train as colab, official_common as common


class ColabTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((Path(__file__).resolve().parents[1] / "config/colab.example.json").read_text())
        self.config.update(clip_snapshot="/content/hf/clip/revision", vocoder_snapshot="/content/hf/vocoder/revision",
            drive_bundle_ready="/content/drive/MyDrive/project/bundle/BUNDLE_READY.json", bundle_sha256="a" * 64,
            smoke_backup="/content/drive/MyDrive/project/backups/smoke_v1", train_backup="/content/drive/MyDrive/project/backups/train_v1")
        self.path = self.root / "config.json"
        common.atomic_json(self.path, self.config)

    def test_unfilled_example_and_drive_active_training_are_rejected(self):
        for key, value in [("bundle_sha256", None), ("smoke_run", "/content/drive/MyDrive/run"),
                           ("train_backup", "/content/local_backup"), ("drive_mount", "/content")]:
            with self.subTest(key=key), self.assertRaises((ValueError, TypeError)):
                colab.validate_config({**self.config, key: value})

    def test_preview_never_copies_runs_or_starts_subprocesses(self):
        with patch.object(subprocess, "run", side_effect=AssertionError("No processes")), patch.object(colab, "stage_bundle", side_effect=AssertionError("No copies")):
            result = colab.run(self.path, "prepare")
        self.assertEqual(result["status"], "COLAB_PREVIEW")
        self.assertFalse(result["gpu_started"])

    def test_colab_execute_cannot_run_on_mac_or_without_drive_mount(self):
        with patch("sys.platform", "darwin"), self.assertRaisesRegex(ValueError, "inside Colab"):
            colab.run(self.path, "prepare", True)
        with patch("sys.platform", "linux"), patch.object(Path, "is_dir", return_value=True), patch("os.path.ismount", return_value=False):
            with self.assertRaisesRegex(ValueError, "mounted"):
                colab.colab_environment(self.config)

    def test_formal_training_is_explicit_and_session_bounded(self):
        with self.assertRaisesRegex(ValueError, "target updates"):
            colab.commands(self.config, "train")
        self.config["recipe"]["steps"] = 1000
        command = colab.commands(self.config, "train")[0]
        self.assertEqual(command[command.index("--steps") + 1], "1000")
        self.assertEqual(command[command.index("--stop-after") + 1], "250")
        self.assertEqual(command[command.index("--backup-dir") + 1], self.config["train_backup"])

    def test_final_session_does_not_stop_before_final_ema(self):
        self.config["recipe"]["steps"] = 500
        self.config["session_updates"] = 500
        self.assertNotIn("--stop-after", colab.commands(self.config, "train")[0])

    def test_smoke_is_separate_and_backs_up_both_processes(self):
        commands = colab.commands(self.config, "smoke")
        self.assertEqual(len(commands), 2)
        self.assertIn("--stop-after", commands[0])
        self.assertIn("--resume", commands[1])
        self.assertTrue(all(self.config["smoke_backup"] in c for c in commands))
        self.assertTrue(all("--execute" not in c for c in commands))

    def test_nested_runs_and_bindings_are_rejected(self):
        for key, value in [("train_run", self.config["smoke_run"] + "/nested"),
                           ("binding", self.config["train_run"] + "/binding.json")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                colab.validate_config({**self.config, key: value})

    def test_smoke_restores_only_persisted_state_before_second_process(self):
        from fine_tune import run_snapshot
        for key in ("smoke_run", "train_run", "smoke_backup", "train_backup"):
            self.config[key] = str((self.root / key).resolve())
        common.atomic_json(self.path, self.config)
        run = Path(self.config["smoke_run"])
        calls = []
        recipe = {"synthetic": True}

        def fake_training(command, **kwargs):
            calls.append(command)
            resumed = "--resume" in command
            step = 24 if resumed else 20
            if not resumed:
                (run / "ema_ckpts").mkdir(parents=True)
                common.atomic_json(run / "run_metadata.json", dict(recipe=recipe))
                common.atomic_json(run / "effective_config.json", {})
            else:
                self.assertFalse((run / "not_backed_up.txt").exists())
                self.assertTrue(run.with_name(run.name + ".before-restore").is_dir())
                self.assertEqual(colab.read_json(run / "latest_checkpoint.json")["completed_updates"], 20)
                common.atomic_json(run / "smoke_report.json", dict(status="PASSED", checkpoint_roundtrip_verified=True))
            checkpoint = run / "checkpoint_slot_1.pth"
            checkpoint.write_bytes(f"synthetic state {step}".encode())
            receipt = dict(path=str(checkpoint), sha256=common.sha256_file(checkpoint), completed_updates=step, recipe=recipe)
            common.atomic_json(checkpoint.with_suffix(".json"), receipt)
            common.atomic_json(run / "latest_checkpoint.json", receipt)
            common.atomic_json(run / "training_status.json", dict(phase="complete" if resumed else "paused", completed_updates=step))
            run_snapshot.save(run, self.config["smoke_backup"])
            if not resumed:
                (run / "not_backed_up.txt").touch()
            return subprocess.CompletedProcess(command, 0)

        with patch.object(colab, "validate_config", side_effect=lambda x: x), patch.object(colab, "colab_environment"), patch.object(subprocess, "run", side_effect=fake_training):
            result = colab.run(self.path, "smoke", True)
        self.assertEqual(result["completed_updates"], 24)
        self.assertEqual(len(calls), 2)
        self.assertEqual(colab.read_json(Path(self.config["smoke_backup"]) / "COLAB_SMOKE_READY.json")["status"], "DRIVE_RESTORE_SMOKE_PASSED")

    def test_notebook_default_cells_are_valid_and_have_no_side_effects(self):
        import ast
        import contextlib
        import io
        from scripts.build_colab_training_notebook import build
        notebook = build()
        self.assertEqual(notebook["nbformat"], 4)
        namespace = {}
        with patch.object(subprocess, "run", side_effect=AssertionError("No process expected")), patch.object(subprocess, "check_output", side_effect=AssertionError("No process expected")), contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse(cell["source"])
                    exec(compile(cell["source"], "<notebook-preview>", "exec"), namespace)
        self.assertFalse(namespace["EXECUTE"])
        self.assertIsNone(namespace["CONFIG_PATH"])
