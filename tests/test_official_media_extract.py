"""CPU/synthetic contracts for caption-independent official media extraction.

These tests do not run a decoder, encoder, GPU, or approve source content.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from fine_tune import official_media_extract as media
from fine_tune.official_common import MODEL_SPECS, NORMALIZE_AUDIO, PINNED_COMMIT
from fine_tune.official_extract import row_paths
from fine_tune.official_extract import fingerprint


SPEC = MODEL_SPECS["small_44k"]
KEYS = {"mean", "std", "clip_features", "sync_features"}
SHAPES = {"mean": (345, 40), "std": (345, 40),
          "clip_features": (64, 1024), "sync_features": (192, 768)}


class FakeTensor:
    """Only the public validation surface, not fake numerical inference."""
    def __init__(self, shape, *, dtype="torch.float32", low=0.1,
                 finite=True, nonzero=True, floating=True):
        self.shape = shape
        self.dtype = dtype
        self.low = low
        self.finite = finite
        self.nonzero = nonzero
        self.floating = floating

    def is_floating_point(self):
        return self.floating

    def isfinite(self):
        return SimpleNamespace(all=lambda: self.finite)

    def ne(self, _):
        return SimpleNamespace(any=lambda: self.nonzero)

    def lt(self, number):
        return SimpleNamespace(any=lambda: self.low < number)


def values():
    return {key: FakeTensor(shape) for key, shape in SHAPES.items()}


def accepted(count=126):
    rows = []
    for index in range(count):
        split = ("train", "val", "test")[index % 3]
        rows.append({
            "training_id": f"dataset__recording_{index:03d}__000000",
            "clip_id": f"recording_{index:03d}__000000",
            "split": split,
            "source_dataset_id": "dataset",
            "source_file": f"recording_{index:03d}.mov",
            "source_start_seconds": "8",
            "media_sha256": hashlib.sha256(f"media{index}".encode()).hexdigest(),
            "sha256": hashlib.sha256(f"media{index}".encode()).hexdigest(),
            "byte_size": str(1000 + index),
            "absolute_path": f"/unread/synthetic/{split}/clip{index}.mp4",
            "caption": "A pending, unapproved draft.",
            "label": "A pending, unapproved draft.",
            "manual_review_status": "pending_review",
            "clapboard_final_status": "included",
            "content_review_status": "not_completed",
            "official_decoder_status": "pass",
            "official_decoder_validated": "true",
            "official_decoder_binding_sha256": "a" * 64,
        })
    return rows


def plan(count=126):
    return {
        "rows": media.media_rows(accepted(count)),
        "model": "small_44k",
        "spec": dict(SPEC),
        "normalize_audio": dict(NORMALIZE_AUDIO),
        "official_commit": PINNED_COMMIT,
        "accepted_manifest_sha256": "b" * 64,
        "official_audit_binding_sha256": "a" * 64,
    }


def validated_plan(count=126):
    result = plan(count)
    result.update(scope="official_media_only_v1", training_ready=False,
                  counts={split: sum(r["split"] == split for r in result["rows"])
                          for split in NORMALIZE_AUDIO})
    result["media_plan_id"] = fingerprint(media.media_identity(result))
    return result


class MediaIdentityTests(unittest.TestCase):
    def test_projection_keeps_global_and_split_row_identity(self):
        source = accepted(9)
        rows = media.media_rows(source)
        self.assertEqual([row["clip_id"] for row in rows],
                         [row["training_id"] for row in source])
        self.assertEqual([row["manifest_row"] for row in rows], list(range(9)))
        self.assertEqual([row["split_row"] for row in rows], [0, 0, 0, 1, 1, 1, 2, 2, 2])
        self.assertEqual(rows[0]["source_recording_id"], "dataset::recording_000.mov")
        self.assertEqual(rows[0]["sha256"], source[0]["media_sha256"])
        self.assertEqual(int(rows[0]["byte_size"]), 1000)

    def test_caption_changes_do_not_change_projected_media_identity(self):
        original = accepted(9)
        changed = deepcopy(original)
        for row in changed:
            row["caption"] = "A reviewed replacement caption."
            row["label"] = "Different text."
            row["caption_final"] = "Final text."
        first, second = plan(9), plan(9)
        first["rows"], second["rows"] = media.media_rows(original), media.media_rows(changed)
        self.assertEqual(media.media_identity(first), media.media_identity(second))

    def test_location_and_caption_metadata_do_not_change_identity(self):
        original = plan(9)
        changed = deepcopy(original)
        changed["plan_path"] = "/a/new/location/plan.json"
        changed["caption_manifest_sha256"] = "c" * 64
        for row in changed["rows"]:
            row["path"] = "/relocated/" + row["clip_id"] + ".mp4"
            row["caption"] = "Caption may change before text features are frozen."
            row["label"] = "Unused metadata."
        self.assertEqual(media.media_identity(original), media.media_identity(changed))

    def test_identity_binds_order_media_split_and_source(self):
        original = plan(9)
        for key, replacement in (("sha256", "d" * 64), ("byte_size", 9999),
                                 ("split", "test"), ("split_row", 7),
                                 ("manifest_row", 7), ("source_recording_id", "different-source"),
                                 ("source_start_seconds", "16"), ("clip_id", "other-id")):
            changed = deepcopy(original)
            changed["rows"][0][key] = replacement
            with self.subTest(key=key):
                self.assertNotEqual(media.media_identity(original), media.media_identity(changed))
        changed = deepcopy(original)
        changed["rows"].reverse()
        self.assertNotEqual(media.media_identity(original), media.media_identity(changed))

    def test_identity_binds_spec_normalization_and_audited_provenance(self):
        original = plan(9)
        changes = [
            ("spec", {**SPEC, "audio_samples": 352800}),
            ("normalize_audio", {**NORMALIZE_AUDIO, "train": False}),
            ("official_commit", "c" * 40),
            ("accepted_manifest_sha256", "c" * 64),
            ("official_audit_binding_sha256", "c" * 64),
            ("model", "small_16k"),
        ]
        for key, replacement in changes:
            changed = deepcopy(original)
            changed[key] = replacement
            with self.subTest(key=key):
                self.assertNotEqual(media.media_identity(original), media.media_identity(changed))


class MediaPlanGuardsTests(unittest.TestCase):
    def test_valid_media_plan_never_implies_caption_or_training_approval(self):
        frozen = validated_plan(9)
        frozen["caption_status"] = "UNCHANGED_NOT_APPROVED"
        frozen["content_review_status"] = "UNCHANGED"
        self.assertEqual(media.validate_plan(frozen), frozen)
        for key, replacement in (("training_ready", True), ("scope", "full_extraction")):
            invalid = deepcopy(frozen)
            invalid[key] = replacement
            with self.subTest(key=key), self.assertRaises(ValueError):
                media.validate_plan(invalid)

    def test_plan_refuses_changed_rows_before_attempting_any_media_reads(self):
        frozen = validated_plan(9)
        changed = deepcopy(frozen)
        changed["rows"][0]["sha256"] = "c" * 64
        with self.assertRaises(ValueError):
            media.validate_plan(changed)

    def test_rehashed_plan_still_rejects_cross_split_source_and_duplicate_ids(self):
        frozen = validated_plan(9)
        for key in ("source_recording_id", "clip_id", "sha256"):
            changed = deepcopy(frozen)
            changed["rows"][1][key] = changed["rows"][0][key]
            changed["media_plan_id"] = fingerprint(media.media_identity(changed))
            with self.subTest(key=key), self.assertRaises(ValueError):
                media.validate_plan(changed)


class MediaSelectionTests(unittest.TestCase):
    def test_six_probe_is_two_per_split_and_subset_of_hundred(self):
        source = plan(126)
        probe = media.select_rows(source, stage="probe6")
        pilot = media.select_rows(source, stage="pilot100")
        self.assertEqual(len(probe), 6)
        self.assertEqual(len(pilot), 100)
        self.assertEqual(pilot[:6], probe)
        self.assertEqual({split: sum(row["split"] == split for row in probe)
                          for split in NORMALIZE_AUDIO}, {"train": 2, "val": 2, "test": 2})
        for split in NORMALIZE_AUDIO:
            self.assertEqual([row["split_row"] for row in probe if row["split"] == split], [0, 1])
        self.assertEqual(len({row["clip_id"] for row in pilot}), 100)
        self.assertEqual(probe, media.select_rows(source, stage="probe6"))

    def test_probe_prefers_two_distinct_recordings_per_split(self):
        source = plan(18)
        # The first two clips in each split are neighbouring clips from one
        # recording. A later clip provides a distinct recording to test.
        for split in NORMALIZE_AUDIO:
            members = [row for row in source["rows"] if row["split"] == split]
            members[1]["source_recording_id"] = members[0]["source_recording_id"]
        probe = media.select_rows(source, stage="probe6")
        self.assertEqual(len({r["source_recording_id"] for r in probe}), 6)
        for split in NORMALIZE_AUDIO:
            self.assertEqual([r["split_row"] for r in probe if r["split"] == split], [0, 2])
        self.assertEqual(media.select_rows(source, stage="pilot100")[:6], probe)

    def test_probe_falls_back_to_two_clips_when_only_one_recording_exists(self):
        source = plan(9)
        for row in source["rows"]:
            row["source_recording_id"] = row["split"] + "-only-source"
        probe = media.select_rows(source, stage="probe6")
        self.assertEqual(len(probe), 6)
        self.assertEqual(len({r["clip_id"] for r in probe}), 6)
        self.assertEqual(len({r["source_recording_id"] for r in probe}), 3)

    def test_batches_cover_once_and_reuse_probe_and_pilot_in_first_batch(self):
        source = plan(1230)
        batches = [media.select_rows(source, stage="batch", batch_index=index, batch_size=500)
                   for index in range(3)]
        self.assertEqual([len(batch) for batch in batches], [500, 500, 230])
        flattened = [row["clip_id"] for batch in batches for row in batch]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(set(flattened), {row["clip_id"] for row in source["rows"]})
        self.assertEqual(batches[0][:100], media.select_rows(source, stage="pilot100"))
        # Neither subset numbering nor batches rewrite the global row contracts.
        indexed = {row["clip_id"]: row for row in source["rows"]}
        self.assertTrue(all(row == indexed[row["clip_id"]] for batch in batches for row in batch))

    def test_hundred_stage_handles_small_fixture_without_duplicate_rows(self):
        selected = media.select_rows(plan(9), stage="pilot100")
        self.assertEqual(len(selected), 9)
        self.assertEqual(len({row["clip_id"] for row in selected}), 9)

    def test_negative_or_empty_batch_size_is_rejected(self):
        for index, size in ((-1, 500), (0, 0), (0, -1)):
            with self.subTest(index=index, size=size), self.assertRaises(ValueError):
                media.select_rows(plan(9), stage="batch", batch_index=index, batch_size=size)


class MediaPreviousStageTests(unittest.TestCase):
    @staticmethod
    def write_completion(output, frozen, previous):
        selected = media.select_rows(frozen, previous)
        selection = fingerprint([row["clip_id"] for row in selected])
        report = {
            "status": "MEDIA_SELECTION_COMPLETE", "run_id": "test-run",
            "selection_id": selection, "selected": len(selected), "checked": len(selected),
            "rows": [row["clip_id"] for row in selected], "training_ready": False,
        }
        path = output / f"{previous}_000_{selection[:12]}_COMPLETE.json"
        path.write_text(json.dumps(report))
        return selected, path, report

    def test_probe_does_not_require_any_prior_stage(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(media, "load_cached") as load:
            media.require_previous_stage(Path(temporary), plan(126), "probe6", "test-run", SPEC, None)
            load.assert_not_called()

    def test_pilot_requires_probe_and_batch_requires_pilot_not_just_probe(self):
        frozen = plan(126)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with self.assertRaises(ValueError):
                media.require_previous_stage(output, frozen, "pilot100", "test-run", SPEC, None)
            self.write_completion(output, frozen, "probe6")
            with self.assertRaises(ValueError):
                media.require_previous_stage(output, frozen, "batch", "test-run", SPEC, None)

    def test_matching_receipt_rechecks_every_prior_cache(self):
        frozen = plan(126)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            for stage, previous in (("pilot100", "probe6"), ("batch", "pilot100")):
                selected, _, _ = self.write_completion(output, frozen, previous)
                with self.subTest(stage=stage), patch.object(media, "load_cached") as load:
                    media.require_previous_stage(output, frozen, stage, "test-run", SPEC, None)
                    self.assertEqual(load.call_count, len(selected))
                    for call, row in zip(load.call_args_list, selected):
                        self.assertEqual(call.args[:5],
                                         (output / "row_cache" / row["split"], row["split_row"],
                                          row, "test-run", SPEC))

    def test_stale_or_incomplete_prior_report_is_rejected_before_cache_reads(self):
        frozen = plan(126)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            _, path, report = self.write_completion(output, frozen, "probe6")
            mutations = (("status", "INCOMPLETE"), ("run_id", "other-run"),
                         ("selection_id", "c" * 64), ("selected", 5),
                         ("checked", 5), ("rows", report["rows"][:-1]))
            for key, replacement in mutations:
                path.write_text(json.dumps({**report, key: replacement}))
                with self.subTest(key=key), patch.object(media, "load_cached") as load:
                    with self.assertRaises(ValueError):
                        media.require_previous_stage(output, frozen, "pilot100", "test-run", SPEC, None)
                    load.assert_not_called()

    def test_valid_summary_cannot_mask_invalid_cache_receipt_or_bytes(self):
        frozen = plan(126)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            self.write_completion(output, frozen, "probe6")
            with patch.object(media, "load_cached", side_effect=ValueError("changed prior feature bytes")):
                with self.assertRaisesRegex(ValueError, "changed prior feature"):
                    media.require_previous_stage(output, frozen, "pilot100", "test-run", SPEC, None)


class MediaPortableSourceTests(unittest.TestCase):
    def test_original_and_portable_locations_resolve_same_clip_identity(self):
        row = plan(9)["rows"][0]
        self.assertEqual(media.source_path(row), Path(row["path"]))
        self.assertEqual(media.source_path(row, Path("/content/staged/media")),
                         Path("/content/staged/media/train") / (row["clip_id"] + ".mp4"))
        changed = {**row, "path": "/obsolete/local/path.mp4"}
        self.assertEqual(media.source_path(changed, "/content/staged/media"),
                         media.source_path(row, "/content/staged/media"))

    def test_portable_copy_still_requires_exact_manifest_bytes(self):
        row = plan(9)["rows"][0]
        with tempfile.TemporaryDirectory() as temporary:
            destination = media.source_path(row, temporary)
            destination.parent.mkdir(parents=True)
            original = b"real synthetic media bytes"
            destination.write_bytes(original)
            row.update(byte_size=len(original), sha256=hashlib.sha256(original).hexdigest())
            media.verify_media(destination, row)
            # Same size but different bytes must not be accepted as a valid copy.
            destination.write_bytes(b"X" + original[1:])
            with self.assertRaises(ValueError):
                media.verify_media(destination, row)
            destination.write_bytes(original + b"extra")
            with self.assertRaises(ValueError):
                media.verify_media(destination, row)


class MediaFeatureTests(unittest.TestCase):
    def test_exact_four_modalities_are_accepted_without_text(self):
        features = values()
        self.assertEqual(set(features), KEYS)
        media.validate_media_features(features, SPEC)

    def test_missing_or_added_text_features_are_refused(self):
        for mutation in ("missing", "text"):
            features = values()
            if mutation == "missing":
                del features["sync_features"]
            else:
                features["text_features"] = FakeTensor((77, 1024))
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                media.validate_media_features(features, SPEC)

    def test_shape_finiteness_float32_nonzero_and_std_guards(self):
        mutations = [
            ("mean", FakeTensor((250, 20))),
            ("clip_features", FakeTensor((64, 1024), finite=False)),
            ("sync_features", FakeTensor((192, 768), nonzero=False)),
            ("mean", FakeTensor((345, 40), dtype="torch.float16")),
            ("mean", FakeTensor((345, 40), floating=False)),
            ("std", FakeTensor((345, 40), low=-0.1)),
        ]
        for key, replacement in mutations:
            features = values()
            features[key] = replacement
            with self.subTest(key=key, replacement=replacement.__dict__), self.assertRaises(ValueError):
                media.validate_media_features(features, SPEC)

    def test_committed_cache_binds_media_row_and_feature_hash(self):
        row = plan(9)["rows"][0]
        features = values()
        fake_torch = SimpleNamespace(float32="torch.float32",
                                     load=lambda *args, **kwargs: features)
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            data_path, receipt_path = row_paths(cache, row["split_row"])
            data_path.write_bytes(b"synthetic feature container")
            receipt = {
                "status": "MEDIA_COMPLETE", "run_id": "test-run",
                "row_index": row["split_row"], "clip_id": row["clip_id"],
                "split": row["split"], "manifest_row": row["manifest_row"],
                "source_recording_id": row["source_recording_id"],
                "source_sha256": row["sha256"],
                "feature_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
            }
            receipt_path.write_text(json.dumps(receipt))
            self.assertEqual(media.load_cached(cache, row["split_row"], row, "test-run", SPEC, fake_torch), features)
            # Caption metadata is irrelevant; it must not invalidate valid media.
            changed_caption = {**row, "label": "changed", "caption": "new draft"}
            media.load_cached(cache, row["split_row"], changed_caption, "test-run", SPEC, fake_torch)
            for key, bad in (("run_id", "other-run"), ("row_index", 5),
                             ("clip_id", "other-clip"), ("split", "test"),
                             ("manifest_row", 5), ("source_recording_id", "other-source"),
                             ("source_sha256", "d" * 64), ("status", "COMPLETE")):
                receipt_path.write_text(json.dumps({**receipt, key: bad}))
                with self.subTest(key=key), self.assertRaises(ValueError):
                    media.load_cached(cache, row["split_row"], row, "test-run", SPEC, fake_torch)
            receipt_path.write_text(json.dumps(receipt))
            data_path.write_bytes(b"changed container")
            with self.assertRaises(ValueError):
                media.load_cached(cache, row["split_row"], row, "test-run", SPEC, fake_torch)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional real CPU round-trip needs torch")
    def test_real_cpu_cache_readback_float16_rejection_and_byte_tamper(self):
        import torch
        row = plan(9)["rows"][0]
        tensors = {key: torch.full(shape, 0.25, dtype=torch.float32, device="cpu")
                   for key, shape in SHAPES.items()}
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            data_path, receipt_path = row_paths(cache, row["split_row"])
            receipt = {
                "status": "MEDIA_COMPLETE", "run_id": "test-run",
                "row_index": row["split_row"], "clip_id": row["clip_id"],
                "split": row["split"], "manifest_row": row["manifest_row"],
                "source_recording_id": row["source_recording_id"],
                "source_sha256": row["sha256"],
            }

            def save(values):
                torch.save(values, data_path)
                receipt["feature_sha256"] = hashlib.sha256(data_path.read_bytes()).hexdigest()
                receipt_path.write_text(json.dumps(receipt))

            save(tensors)
            loaded = media.load_cached(cache, row["split_row"], row, "test-run", SPEC, torch)
            self.assertEqual(set(loaded), KEYS)
            for key in KEYS:
                self.assertEqual(loaded[key].device.type, "cpu")
                self.assertEqual(loaded[key].dtype, torch.float32)
                self.assertTrue(torch.equal(loaded[key], tensors[key]))

            save({**tensors, "mean": tensors["mean"].half()})
            with self.assertRaisesRegex(ValueError, "shape/dtype"):
                media.load_cached(cache, row["split_row"], row, "test-run", SPEC, torch)

            save(tensors)
            data_path.write_bytes(data_path.read_bytes() + b"unauthorised trailing bytes")
            with self.assertRaisesRegex(ValueError, "hash"):
                media.load_cached(cache, row["split_row"], row, "test-run", SPEC, torch)


if __name__ == "__main__":
    unittest.main()
