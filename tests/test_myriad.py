import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from fine_tune import myriad
from fine_tune import official_common as common, official_train as train


class MyriadTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = json.loads((Path(__file__).resolve().parents[1] / "config/myriad.example.json").read_text())
        for key in ("code_root", "python", "official_repo", "weights_dir", "clip_snapshot", "vocoder_snapshot",
                    "pretrained", "bundle_ready", "binding", "smoke_run", "train_run", "job_output"):
            self.config[key] = str(self.root / key)
        self.config["bundle_sha256"] = "a" * 64
        self.config["bundle_ready"] = str(self.root / "bundle/BUNDLE_READY.json")
        self.path = self.root / "config.json"
        common.atomic_json(self.path, self.config)

    def test_unfilled_example_is_blocked(self):
        value = {**self.config, "python": None}
        with self.assertRaisesRegex(ValueError, "absolute path"):
            myriad.validate_config(value, concrete=True)

    def test_render_preview_does_not_submit_or_write(self):
        output = self.root / "scripts"
        with patch.object(subprocess, "run", side_effect=AssertionError("No process expected")):
            result = myriad.render(self.path, output)
        self.assertEqual(result["status"], "RENDER_PREVIEW")
        self.assertFalse(output.exists())

    def test_rendered_scripts_parse_and_request_per_core_memory(self):
        output = self.root / "scripts"
        myriad.render(self.path, output, True)
        for path in output.glob("*.qsub"):
            subprocess.run(["bash", "-n", str(path)], check=True)
            text = path.read_text()
            self.assertIn("#$ -l mem=8G", text)
            self.assertIn("#$ -pe smp 4", text)
            self.assertIn("JOB_ID:?", text)
            self.assertNotIn("sbatch", text)
            self.assertNotIn("module load cuda/", text)

    def test_login_node_execution_is_rejected(self):
        with patch.dict("os.environ", {}, clear=True), self.assertRaisesRegex(ValueError, "SGE job"):
            myriad.job(self.path, common.sha256_file(self.path), "verify", True)

    def test_editing_config_requires_new_scripts(self):
        digest = common.sha256_file(self.path)
        self.path.write_text(self.path.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            myriad.job(self.path, digest, "verify")

    def test_formal_training_has_no_implicit_target(self):
        with self.assertRaisesRegex(ValueError, "target updates"):
            myriad.training_command(self.config, "train")
        command = myriad.training_command(self.config, "smoke")
        self.assertIn("--checkpoint-probe", command)
        self.assertIn("24", command)
        self.assertNotIn("--execute", command)

    def test_resource_injection_is_rejected(self):
        for key in ("mem_per_core", "tmpfs", "walltime"):
            value = copy.deepcopy(self.config)
            value["resources"][key] = "8G\n#$ -l gpu=4"
            with self.subTest(key=key), self.assertRaises(ValueError):
                myriad.validate_config(value)

    def test_runtime_outputs_cannot_pollute_bundle_or_runs(self):
        for key, path in [("binding", self.root / "bundle/binding.json"),
                          ("job_output", self.root / "train_run/jobs"),
                          ("train_run", self.root / "smoke_run/train")]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                myriad.validate_config({**self.config, key: str(path)}, concrete=True)

    def test_smoke_job_requires_a_second_process_and_successful_reload(self):
        import os
        import sys
        self.config.update(python=sys.executable, scheduler_verified=True)
        common.atomic_json(self.path, self.config)
        calls = []
        run = Path(self.config["smoke_run"])

        def fake_run(command, **kwargs):
            calls.append(command)
            if "--checkpoint-probe" in command:
                if "--stop-after" in command:
                    run.mkdir()
                    checkpoint = run / "checkpoint.pth"
                    checkpoint.write_bytes(b"synthetic checkpoint")
                    common.atomic_json(run / "latest_checkpoint.json", dict(path=str(checkpoint), sha256=common.sha256_file(checkpoint)))
                    common.atomic_json(run / "training_status.json", dict(phase="paused", completed_updates=20))
                else:
                    self.assertIn("--resume", command)
                    common.atomic_json(run / "training_status.json", dict(phase="completed", completed_updates=24))
                    common.atomic_json(run / "smoke_report.json", dict(status="PASSED", checkpoint_roundtrip_verified=True))
            return subprocess.CompletedProcess(command, 0, "fixture", "")

        with patch.dict(os.environ, dict(JOB_ID="123", NSLOTS="4", SGE_O_WORKDIR=str(self.root))), patch.object(subprocess, "run", side_effect=fake_run):
            result = myriad.job(self.path, common.sha256_file(self.path), "smoke", True)
        self.assertEqual(result["phase"], "completed")
        self.assertEqual(sum("--checkpoint-probe" in c for c in calls), 2)
        self.assertEqual(sum("fine_tune.official_preflight" in c for c in calls), 1)

    def test_checkpoint_state_comparison_detects_changes(self):
        import torch
        a = {"optimizer": {0: torch.ones(4)}, "rng": (1, [2, 3])}
        b = copy.deepcopy(a)
        self.assertTrue(train.equal_state(a, b, torch))
        b["optimizer"][0][0] = 2
        self.assertFalse(train.equal_state(a, b, torch))

    def test_transfer_script_defaults_to_immutable_copy_dry_run(self):
        import os
        script = Path(__file__).resolve().parents[1] / "scripts/transfer_feature_bundle.sh"
        rclone = self.root / "rclone"
        rclone.write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$RCLONE_ARGS"\n')
        rclone.chmod(0o755)
        args_file = self.root / "args.txt"
        env = {**os.environ, "PATH": str(self.root) + os.pathsep + os.environ["PATH"], "RCLONE_ARGS": str(args_file)}
        destination = self.root / "destination"
        subprocess.run(["bash", str(script), "fixture:bundle", str(destination), "a" * 64], env=env, check=True, capture_output=True)
        args = args_file.read_text().splitlines()
        self.assertEqual(args[:3], ["copy", "fixture:bundle", str(destination)])
        self.assertIn("--immutable", args)
        self.assertIn("--dry-run", args)
        self.assertFalse(destination.exists())

    def test_transfer_refuses_non_absolute_destination_before_network(self):
        script = Path(__file__).resolve().parents[1] / "scripts/transfer_feature_bundle.sh"
        result = subprocess.run(["bash", str(script), "fixture:bundle", "relative", "a" * 64], capture_output=True)
        self.assertEqual(result.returncode, 2)

    def test_bundle_smoke_requires_reload_evidence(self):
        identity = dict(checkpoint_roundtrip_required=True)
        report = dict(status="PASSED", recipe=identity, completed_updates=24,
            optimizer_covers_all_trainable=True, finite_losses_and_gradients=True,
            updated_tensor_fraction=1.0, modalities_verified=True)
        path = self.root / "smoke.json"
        common.atomic_json(path, report)
        with self.assertRaisesRegex(ValueError, "reload"):
            train.verify_smoke(path, identity)


if __name__ == "__main__":
    unittest.main()
