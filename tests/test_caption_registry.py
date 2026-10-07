from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_caption_registry import (
    export_registry, init_registry, load_registry, validate_files,
)


class CaptionRegistryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manifest = self.root / "qc.csv"
        self.registry = self.root / "registry_v1.json"
        self.media = self.root / "unchanged.mp4"
        self.media.write_bytes(b"not real media; this utility must never decode it")
        self.media_digest = hashlib.sha256(self.media.read_bytes()).hexdigest()
        self.rows = [self.row("clip1", "source1", "train"), self.row("clip2", "source2", "val")]
        self.write_manifest()

    def row(self, clip, source, split):
        return {"clip_id": clip, "training_id": "isd__" + clip, "source_recording_id": source,
                "site_id": source + "_site", "source_start_seconds": "8", "duration_seconds": "8.0",
                "split": split, "absolute_path": str(self.media), "manual_review_status": "included",
                "qc_status": "pass", "clapboard_final_status": "included", "sync_sha256": self.media_digest,
                "sync_status": "copied", "qc_audio_length_checked": "True", "qc_audio_samples": "353280",
                "qc_sample_rate": "44100", "label": "stale generic label", "caption": "Distant traffic.",
                "extra_qc_field": "preserve me"}

    def write_manifest(self):
        with self.manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(self.rows[0]))
            writer.writeheader()
            writer.writerows(self.rows)

    def init(self):
        init_registry(self.manifest, "captions_v1", self.registry)
        return load_registry(self.registry)[1]

    def save(self, registry):
        self.registry.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    def approved(self):
        registry = self.init()
        for record in registry["records"]:
            record.update(caption_final="Continuous distant traffic with occasional birdsong.",
                          caption_review_status="approved", reviewer="test reviewer",
                          reviewed_at="2026-09-16T12:00:00+01:00", acoustic_only_reviewed=True)
        self.save(registry)
        return registry

    def validate(self, **kwargs):
        return validate_files(self.manifest, self.registry, "captions_v1", **kwargs)

    def test_init_never_approves_or_invents_perception(self):
        registry = self.init()
        first = registry["records"][0]
        self.assertEqual(first["canonical_id"], "isd__clip1")
        self.assertEqual(first["caption_candidate"], "Distant traffic.")
        self.assertEqual(first["caption_final"], "")
        self.assertEqual(first["caption_review_status"], "pending")
        self.assertFalse(first["acoustic_only_reviewed"])
        self.assertEqual(first["perception"]["scores"], {})
        self.assertEqual(self.validate()["status"], "WAITING_FOR_CAPTION_REVIEW")
        with self.assertRaisesRegex(ValueError, "not approved"):
            self.validate(require_approved=True)

    def test_canonical_id_falls_back_to_clip_id(self):
        for row in self.rows:
            row["training_id"] = ""
        self.write_manifest()
        self.assertEqual(self.init()["records"][0]["canonical_id"], "clip1")

    def test_approved_export_preserves_all_fields_order_proposal_and_media(self):
        registry = self.approved()
        # Annotation row order may differ; output always follows original manifest order.
        registry["records"].reverse()
        self.save(registry)
        original = self.manifest.read_bytes()
        output = self.root / "b0_v1"
        summary = export_registry(self.manifest, self.registry, "captions_v1", "b0_v1", output)
        with (output / "captions_manifest.csv").open(newline="") as handle:
            result = list(csv.DictReader(handle))
        for before, after in zip(self.rows, result):
            for field in before:
                if field not in {"label", "caption"}:
                    self.assertEqual(after[field], before[field])
            self.assertEqual(after["label"], after["caption"])
            self.assertNotEqual(after["label"], before["label"])
        self.assertEqual(self.manifest.read_bytes(), original)
        self.assertEqual(hashlib.sha256(self.media.read_bytes()).hexdigest(), self.media_digest)
        snapshot = load_registry(output / "registry_snapshot.json")[1]
        self.assertEqual(snapshot["records"][0]["caption_candidate"], "Distant traffic.")
        self.assertEqual(summary["media_files_read_or_changed"], 0)
        self.assertIn("UNVERIFIED", summary["tokenizer_validation"])
        self.assertIn("NOT_ASSESSED", summary["training_readiness"])
        for filename, expected in summary["files_sha256"].items():
            self.assertEqual(hashlib.sha256((output / filename).read_bytes()).hexdigest(), expected)

    def test_init_and_export_dry_run_write_nothing(self):
        pending = self.root / "absent" / "pending.json"
        init_registry(self.manifest, "captions_v1", pending, dry_run=True)
        self.assertFalse(pending.parent.exists())
        self.approved()
        output = self.root / "absent_export" / "b0_v1"
        export_registry(self.manifest, self.registry, "captions_v1", "b0_v1", output, dry_run=True)
        self.assertFalse(output.parent.exists())

    def test_no_overwrite(self):
        self.approved()
        before = self.registry.read_bytes()
        with self.assertRaisesRegex(ValueError, "overwrite"):
            init_registry(self.manifest, "captions_v2", self.registry)
        output = self.root / "b0_v1"
        export_registry(self.manifest, self.registry, "captions_v1", "b0_v1", output)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            export_registry(self.manifest, self.registry, "captions_v1", "b0_v2", output)
        self.assertEqual(self.registry.read_bytes(), before)

    def test_manifest_hash_version_and_identity_tampering_rejected(self):
        registry = self.init()
        with self.assertRaisesRegex(ValueError, "version mismatch"):
            validate_files(self.manifest, self.registry, "captions_v2")
        for field, value in (("split", "test"), ("start_seconds", 99), ("site_id", "wrong"),
                              ("source_recording_id", "another_source")):
            modified = copy.deepcopy(registry)
            modified["records"][0]["identity"][field] = value
            self.save(modified)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                self.validate()
        self.save(registry)
        self.rows[0]["label"] = "Input changed after registry was created"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "SHA256"):
            self.validate()

    def test_duplicate_missing_unknown_registry_ids_rejected(self):
        registry = self.init()
        modified = copy.deepcopy(registry)
        modified["records"].append(modified["records"][0])
        self.save(modified)
        with self.assertRaisesRegex(ValueError, "Duplicate registry ID"):
            self.validate()
        for mode in ("missing", "unknown", "absent_field"):
            modified = copy.deepcopy(registry)
            if mode == "missing":
                modified["records"].pop()
            elif mode == "unknown":
                modified["records"][0]["canonical_id"] = "unknown"
            else:
                del modified["records"][0]["canonical_id"]
            self.save(modified)
            with self.assertRaisesRegex(ValueError, "coverage mismatch|Missing record"):
                self.validate()

    def test_duplicate_or_missing_manifest_ids_rejected(self):
        self.rows[1]["training_id"] = self.rows[0]["training_id"]
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Duplicate canonical ID"):
            self.init()
        self.rows[0]["training_id"] = self.rows[0]["clip_id"] = ""
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "Missing or unsafe"):
            self.init()

    def test_source_split_leak_and_pending_qc_rejected(self):
        self.rows[1]["source_recording_id"] = "source1"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "split leakage"):
            self.init()
        self.rows[1]["source_recording_id"] = "source2"
        self.rows[0]["qc_status"] = "pending"
        self.write_manifest()
        with self.assertRaisesRegex(ValueError, "QC-passed"):
            self.init()

    def test_human_approval_requires_final_reviewer_timestamp_and_acoustic_check(self):
        registry = self.approved()
        for field, value in (("caption_final", ""), ("reviewer", ""), ("reviewed_at", ""),
                              ("reviewed_at", "2026-09-16T12:00:00"),
                              ("acoustic_only_reviewed", False), ("acoustic_only_reviewed", "true")):
            modified = copy.deepcopy(registry)
            modified["records"][0][field] = value
            self.save(modified)
            with self.assertRaises(ValueError):
                self.validate(require_approved=True)

    def test_caption_provenance_control_characters_and_original_values_checked(self):
        registry = self.approved()
        for field, value in (("caption_final", "birds\ntraffic"), ("caption_candidate", "\x00birds"),
                              ("caption_source", "missing"), ("caption_source", "soundscaper"),
                              ("original_caption", {"label": "altered", "caption": "Distant traffic."})):
            modified = copy.deepcopy(registry)
            modified["records"][0][field] = value
            self.save(modified)
            with self.assertRaises(ValueError):
                self.validate()

    def test_model_candidate_requires_model_and_prompt_version(self):
        registry = self.approved()
        record = registry["records"][0]
        record.update(caption_source="soundscaper", caption_candidate="Birdsong and distant traffic.")
        record["generation"] = {"model": "SoundSCaper+local-llm-revision", "prompt_version": "acoustic_v1"}
        self.save(registry)
        self.assertEqual(self.validate()["status"], "READY_FOR_B0_CAPTION_EXPORT")
        record["generation"]["prompt_version"] = ""
        self.save(registry)
        with self.assertRaisesRegex(ValueError, "prompt_version"):
            self.validate()

    def valid_perception(self):
        return {"label_source": "human", "dimension_ids": ["pleasant", "eventful"],
                "scores": {"pleasant": 4, "eventful": 2}, "scale": {"name": "Likert", "min": 1, "max": 5},
                "temporal_scope": {"unit": "clip", "scope_id": "isd__clip1", "start_seconds": 8, "end_seconds": 16},
                "annotation_version": "human_paq_v1", "annotator_or_model": "panel-A", "inherited_from": ""}

    def test_optional_perception_complete_metadata_valid_without_injection(self):
        registry = self.approved()
        registry["records"][0]["perception"] = self.valid_perception()
        self.save(registry)
        self.assertEqual(self.validate()["status"], "READY_FOR_B0_CAPTION_EXPORT")

    def test_perception_partial_wrong_scale_time_and_missing_labels_rejected(self):
        registry = self.init()
        for field, value in (("label_source", "missing"), ("scores", {"pleasant": 10, "eventful": 2}),
                              ("dimension_ids", ["pleasant"]), ("annotation_version", ""),
                              ("scale", None), ("temporal_scope", None)):
            perception = self.valid_perception()
            perception[field] = value
            registry["records"][0]["perception"] = perception
            self.save(registry)
            with self.assertRaises(ValueError):
                self.validate()
        perception = self.valid_perception()
        perception["temporal_scope"]["end_seconds"] = 30
        registry["records"][0]["perception"] = perception
        self.save(registry)
        with self.assertRaisesRegex(ValueError, "temporal scope"):
            self.validate()

    def test_long_recording_labels_must_be_explicitly_inherited(self):
        registry = self.init()
        perception = self.valid_perception()
        perception["temporal_scope"] = {"unit": "source_recording", "scope_id": "source1", "start_seconds": 0, "end_seconds": 100}
        registry["records"][0]["perception"] = perception
        self.save(registry)
        with self.assertRaisesRegex(ValueError, "marked inherited"):
            self.validate()
        perception.update(label_source="inherited", inherited_from="source1-questionnaire-v1")
        self.save(registry)
        self.validate()

    def test_invalid_duration_fingerprint_and_condition_arm_blocked(self):
        for field, value in (("duration_seconds", "nan"), ("source_start_seconds", "-1"),
                              ("sync_sha256", "not-a-sha")):
            old = self.rows[0][field]
            self.rows[0][field] = value
            self.write_manifest()
            with self.assertRaises(ValueError):
                self.init()
            self.rows[0][field] = old
        self.write_manifest()
        registry = self.init()
        registry["condition_arm"] = "text_plus_vector"
        self.save(registry)
        with self.assertRaisesRegex(ValueError, "Only B0"):
            self.validate()

    def test_no_media_access_required_after_manifest_binding(self):
        self.media.unlink()
        self.approved()
        self.assertEqual(self.validate()["media_files_read_or_changed"], 0)

    def test_site_overlap_reported_without_claiming_site_holdout(self):
        self.rows[1]["site_id"] = self.rows[0]["site_id"]
        self.write_manifest()
        self.init()
        summary = self.validate()
        self.assertEqual(summary["site_overlap_across_splits"], {"source1_site": ["train", "val"]})
        self.assertIn("not enforced", summary["split_policy"])

    def test_generation_optional_hash_validated_if_supplied(self):
        registry = self.init()
        registry["records"][0]["generation"]["weights_sha256"] = "unknown"
        self.save(registry)
        with self.assertRaisesRegex(ValueError, "weights_sha256"):
            self.validate()

    def test_declared_annotation_duration_not_container_padding(self):
        for row in self.rows:
            row["actual_duration_seconds"] = "8.011"
        self.write_manifest()
        registry = self.approved()
        first = registry["records"][0]
        first["perception"] = self.valid_perception()
        self.assertEqual(first["identity"]["duration_seconds"], 8.0)
        self.assertEqual(first["identity"]["duration_source_field"], "duration_seconds")
        self.assertEqual(first["identity"]["manifest_fields"]["actual_duration_seconds"], "8.011")
        self.save(registry)
        self.validate()

    def test_dangling_output_symlinks_are_not_usable_even_in_preview(self):
        output = self.root / "dangling"
        output.symlink_to(self.root / "nonexistent")
        for dry_run in (False, True):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                init_registry(self.manifest, "captions_v1", output, dry_run=dry_run)
        self.approved()
        for dry_run in (False, True):
            with self.assertRaisesRegex(ValueError, "overwrite"):
                export_registry(self.manifest, self.registry, "captions_v1", "b0_v1", output, dry_run=dry_run)
        self.assertTrue(output.is_symlink())

    def test_raw_model_candidate_must_be_retained(self):
        registry = self.approved()
        first = registry["records"][0]
        first.update(caption_source="soundscaper", caption_candidate="")
        first["generation"] = {"model": "test-model", "prompt_version": "test-v1"}
        self.save(registry)
        with self.assertRaisesRegex(ValueError, "retained model proposal"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
