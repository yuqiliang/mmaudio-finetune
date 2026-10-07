from pathlib import Path
import tempfile
import unittest

from fine_tune import official_common as c, run_snapshot as s


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.run = self.root / "local/run"
        self.backup = self.root / "drive/backup"
        (self.run / "ema_ckpts").mkdir(parents=True)
        self.recipe = {"model": "fixture", "data_sha256": "a" * 64}
        c.atomic_json(self.run / "run_metadata.json", dict(recipe=self.recipe))
        c.atomic_json(self.run / "effective_config.json", dict(model="fixture"))
        (self.run / "ema_ckpts/20.pt").write_bytes(b"ema history")
        (self.run / ".training.lock").touch()
        (self.run / "STOP_REQUESTED").touch()
        self.checkpoint(20)

    def checkpoint(self, step):
        path = self.run / "checkpoint_slot_1.pth"
        path.write_bytes(f"synthetic checkpoint {step}".encode())
        receipt = dict(path=str(path), sha256=c.sha256_file(path), completed_updates=step, recipe=self.recipe)
        c.atomic_json(path.with_suffix(".json"), receipt)
        c.atomic_json(self.run / "latest_checkpoint.json", receipt)

    def test_roundtrip_preserves_checkpoint_ema_and_metadata(self):
        result = s.save(self.run, self.backup)
        saved, _ = s.verify(result["ready"], result["sha256"])
        self.assertNotIn(".training.lock", saved["files"])
        self.assertNotIn("STOP_REQUESTED", saved["files"])
        self.run.rename(self.root / "original")
        preview = s.restore(result["ready"], result["sha256"], self.run)
        self.assertEqual(preview["status"], "RESTORE_PREVIEW")
        self.assertFalse(self.run.exists())
        s.restore(result["ready"], result["sha256"], self.run, True)
        for name, entry in saved["files"].items():
            self.assertEqual(c.sha256_file(self.run / name), entry["sha256"])
        self.assertTrue((self.run / "ema_ckpts").is_dir())

    def test_unchanged_objects_are_reused_and_previous_snapshot_stays_valid(self):
        first = s.save(self.run, self.backup)
        before = {p.name for p in (self.backup / "objects").iterdir()}
        s.save(self.run, self.backup)
        self.assertEqual(before, {p.name for p in (self.backup / "objects").iterdir()})
        self.checkpoint(24)
        s.save(self.run, self.backup)
        s.verify(first["ready"], first["sha256"])

    def test_corruption_refuses_restore_before_destination_is_created(self):
        result = s.save(self.run, self.backup)
        receipt, _ = s.verify(result["ready"], result["sha256"])
        obj = s.object_path(self.backup, receipt["files"]["checkpoint_slot_1.pth"]["sha256"])
        obj.write_bytes(b"corrupt")
        self.run.rename(self.root / "original")
        with self.assertRaisesRegex(ValueError, "changed"):
            s.restore(result["ready"], result["sha256"], self.run, True)
        self.assertFalse(self.run.exists())

    def test_incomplete_snapshot_cannot_replace_previous_pointer(self):
        s.save(self.run, self.backup)
        pointer = (self.backup / "LATEST.json").read_bytes()
        (self.run / "checkpoint_slot_0.pth.pending").write_bytes(b"partial")
        with self.assertRaisesRegex(ValueError, "Partial"):
            s.save(self.run, self.backup)
        self.assertEqual((self.backup / "LATEST.json").read_bytes(), pointer)

    def test_existing_restore_and_changed_recipe_are_rejected(self):
        result = s.save(self.run, self.backup)
        with self.assertRaisesRegex(ValueError, "exists"):
            s.restore(result["ready"], result["sha256"], self.run, True)
        with self.assertRaisesRegex(ValueError, "same absolute"):
            s.restore(result["ready"], result["sha256"], self.root / "another", True)
        self.recipe["model"] = "changed"
        c.atomic_json(self.run / "run_metadata.json", dict(recipe=self.recipe))
        self.checkpoint(21)
        with self.assertRaisesRegex(ValueError, "another run"):
            s.save(self.run, self.backup)

    def test_snapshot_cannot_roll_back_or_escape_its_directory(self):
        s.save(self.run, self.backup)
        self.checkpoint(19)
        with self.assertRaisesRegex(ValueError, "newer"):
            s.save(self.run, self.backup)
        c.atomic_json(self.backup / "LATEST.json", dict(ready="../elsewhere", sha256="a" * 64))
        with self.assertRaisesRegex(ValueError, "Unsafe"):
            s.latest(self.backup)

    def test_empty_ema_directory_is_recreated_before_first_ema_checkpoint(self):
        (self.run / "ema_ckpts/20.pt").unlink()
        result = s.save(self.run, self.backup)
        self.run.rename(self.root / "original")
        s.restore(result["ready"], result["sha256"], self.run, True)
        self.assertTrue((self.run / "ema_ckpts").is_dir())

    def test_symlinks_and_overlapping_backup_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "overlap"):
            s.save(self.run, self.run / "backup")
        (self.run / "external").symlink_to(self.root / "elsewhere")
        with self.assertRaisesRegex(ValueError, "Symlinks"):
            s.save(self.run, self.backup)
