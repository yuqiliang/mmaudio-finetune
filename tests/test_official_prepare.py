from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from fine_tune.official_common import atomic_json, load_plan, preprocessing_id, sha256_file
from fine_tune.official_prepare import prepare, inventory, write_csv


class OfficialPlanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.review = self.root / "review"
        self.review.mkdir()
        self.rows = []
        for split in ("train", "val", "test"):
            path = self.root / f"{split}.mp4"
            path.write_bytes(f"synthetic-unit-test-only-{split}".encode())
            self.rows.append(dict(training_id=split, clip_id=split, source_file=f"{split}_long.mp4",
                                  source_dataset_id="new", split=split, absolute_path=str(path),
                                  relative_path=path.name, manual_review_status="included", qc_status="pass",
                                  qc_audio_length_checked="True", qc_sample_rate="44100",
                                  qc_audio_samples="353280", source_start_seconds="0",
                                  site_id=f"site_{split}"))
        self.input = self.root / "input.csv"
        self.write_inputs()

    def rewrite_csv(self, path, rows):
        if path.exists():
            path.unlink()
        write_csv(path, rows, list(rows[0]))

    def write_inputs(self):
        self.rewrite_csv(self.input, self.rows)
        self.rewrite_csv(self.review / "final_clips_manifest.csv", self.rows)
        kept = [{"source_dataset_id": "new", "source_file": r["source_file"], "review_decision": "keep"}
                for r in self.rows]
        self.rewrite_csv(self.review / "kept_sources.csv", kept)
        atomic_json(self.review / "summary.json", {"status": "final", "pending_sources": 0,
                    "kept_sources": len(kept), "final_clips": len(self.rows)})

    def run_prepare(self, **kwargs):
        args = dict(manifest=self.input, review_dir=self.review, output=self.root / "plan",
                    dataset_version="expanded_v2", model="small_44k")
        args.update(kwargs)
        return prepare(**args)

    def test_ready_plan_identity_and_nonmutation(self):
        before = {r["absolute_path"]: sha256_file(r["absolute_path"]) for r in self.rows}
        path = self.run_prepare(data_root=self.root)
        plan = load_plan(path)
        self.assertEqual(plan["status"], "READY_FOR_OFFICIAL_EXTRACTION")
        self.assertEqual(plan["preprocessing_id"], preprocessing_id("small_44k"))
        for split in ("train", "val", "test"):
            self.assertEqual(plan["splits"][split]["count"], 1)
            with open(plan["splits"][split]["tsv"]) as handle:
                self.assertEqual(next(csv.reader(handle, delimiter="\t")), ["id", "label"])
        self.assertEqual(before, {p: sha256_file(p) for p in before})
        with self.assertRaisesRegex(ValueError, "overwrite"):
            self.run_prepare()

    def test_interim_review_refused_before_output(self):
        atomic_json(self.review / "summary.json", {"status": "interim", "pending_sources": 1})
        with self.assertRaisesRegex(ValueError, "WAITING_FOR_MANUAL_REVIEW"):
            self.run_prepare()
        self.assertFalse((self.root / "plan").exists())

    def test_unreviewed_or_unchecked_input_rejected(self):
        for key, value in [("manual_review_status", "pending"), ("qc_status", "fail"),
                           ("clapboard_final_status", "pending_review"), ("qc_audio_length_checked", "False")]:
            with self.subTest(key=key):
                rows = [dict(r) for r in self.rows]
                rows[0][key] = value
                self.rewrite_csv(self.input, rows)
                with self.assertRaises(ValueError):
                    self.run_prepare()

    def test_clip_not_in_human_manifest_rejected(self):
        self.rows[0]["training_id"] = "unapproved"
        self.rewrite_csv(self.input, self.rows)
        with self.assertRaisesRegex(ValueError, "not in final"):
            self.run_prepare()

    def test_short_44k_audio_rejected_but_16k_duration_valid(self):
        for r in self.rows:
            r["qc_audio_samples"] = "352800"
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "Audio too short"):
            self.run_prepare()
        self.assertEqual(load_plan(self.run_prepare(model="small_16k"))["model"], "small_16k")

    def test_source_and_site_leakage(self):
        self.rows[1]["source_file"] = self.rows[0]["source_file"]
        self.write_inputs()
        # Duplicate kept-source identity is also a valid fail-closed outcome.
        with self.assertRaises(ValueError):
            self.run_prepare()
        self.rows[1]["source_file"] = "different_long.mp4"
        self.rows[1]["site_id"] = self.rows[0]["site_id"]
        self.write_inputs()
        with self.assertRaisesRegex(ValueError, "site leakage"):
            self.run_prepare(split_policy="site")

    def test_media_modification_invalidates_existing_plan(self):
        path = self.run_prepare()
        Path(self.rows[0]["absolute_path"]).write_bytes(b"changed-media")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_plan(path)

    def test_review_evidence_modification_invalidates_plan(self):
        path = self.run_prepare()
        (path.parent / "review_evidence" / "kept_sources.csv").write_text("modified\n")
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            load_plan(path)

    def test_tsv_reorder_rejected_even_if_hash_updated(self):
        path = self.run_prepare()
        plan = json.loads(path.read_text())
        tsv = Path(plan["splits"]["train"]["tsv"])
        tsv.write_text("id\tlabel\nwrong_id\turban soundscape\n")
        plan["splits"]["train"]["tsv_sha256"] = sha256_file(tsv)
        atomic_json(path, plan)
        with self.assertRaisesRegex(ValueError, "row order"):
            load_plan(path)

    def test_duplicate_bytes_rejected(self):
        Path(self.rows[1]["absolute_path"]).write_bytes(Path(self.rows[0]["absolute_path"]).read_bytes())
        with self.assertRaisesRegex(ValueError, "Identical media"):
            self.run_prepare()

    def test_partial_sync_refused(self):
        for r in self.rows:
            r["sync_status"] = "copied"
            r["sync_sha256"] = sha256_file(r["absolute_path"])
        self.rewrite_csv(self.input, self.rows)
        with self.assertRaisesRegex(ValueError, "sync-summary"):
            self.run_prepare()
        summary = self.root / "sync_summary.json"
        atomic_json(summary, {"dry_run": False, "failed_files": 1, "requested_files": 4,
                              "synced_files": 3, "verification": "sha256"})
        with self.assertRaisesRegex(ValueError, "complete"):
            self.run_prepare(sync_summary=summary)

    def test_complete_synced_existing_layout_needs_no_symlinks(self):
        destination = self.root / "drive"
        synced = []
        for r in self.rows:
            row = dict(r)
            path = destination / r["split"] / f"{r['training_id']}.mp4"
            path.parent.mkdir(parents=True)
            path.write_bytes(Path(r["absolute_path"]).read_bytes())
            row.update(original_absolute_path=r["absolute_path"], absolute_path=str(path),
                       relative_path=str(path.relative_to(destination)), sync_status="replaced",
                       sync_sha256=sha256_file(path))
            synced.append(row)
        self.rewrite_csv(self.input, synced)
        summary = self.root / "sync_summary.json"
        atomic_json(summary, {"dry_run": False, "failed_files": 0, "requested_files": 3,
                              "synced_files": 3, "verification": "sha256"})
        plan_path = self.run_prepare(data_root=destination, sync_summary=summary)
        plan = load_plan(plan_path)
        self.assertEqual(plan["video_layout"], "existing")
        self.assertFalse((plan_path.parent / "videos").exists())
        self.assertEqual(Path(plan["splits"]["train"]["video_root"]), (destination / "train").resolve())

    def test_different_media_cannot_borrow_reviewed_clip_identity(self):
        unrelated = self.root / "unrelated.mp4"
        unrelated.write_bytes(b"different-recording")
        self.rows[0]["absolute_path"] = str(unrelated)
        self.rewrite_csv(self.input, self.rows)
        with self.assertRaisesRegex(ValueError, "Media path changed"):
            self.run_prepare()

    def test_inventory_starts_pending(self):
        folder = self.root / "long_sources"
        folder.mkdir()
        (folder / "example.mov").write_bytes(b"not-real-video")
        result = inventory([folder], self.root / "inventory")
        self.assertEqual(result["status"], "WAITING_FOR_MANUAL_REVIEW")
        self.assertIn("Pending", Path(result["review_file"]).read_text())


if __name__ == "__main__":
    unittest.main()
