"""CPU integration tests: frozen review -> existing caches -> portable memmaps."""
import copy
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from contextlib import ExitStack, nullcontext
import unittest
from unittest.mock import patch

from fine_tune import official_common as c, official_extract as x, official_media_extract as m
from fine_tune import official_feature_bundle as b


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.plan = dict(schema_version=1, scope="official_media_only_v1", training_ready=False,
            model="small_44k", spec=c.MODEL_SPECS["small_44k"], normalize_audio=c.NORMALIZE_AUDIO,
            official_commit=c.PINNED_COMMIT, accepted_manifest_sha256="a" * 64,
            official_audit_binding_sha256="b" * 64, counts={s: 2 for s in c.SPLITS}, rows=[])
        for i, split in enumerate(c.SPLITS):
            for index in range(2):
                self.plan["rows"].append(dict(clip_id=f"{split}_{index}", split=split, split_row=index,
                    manifest_row=i * 2 + index, source_recording_id=f"source_{split}",
                    source_start_seconds=str(index * 8), sha256=f"{i * 2 + index + 1:064x}",
                    byte_size=42, path=f"/old-colab/{split}/{index}.mp4"))
        self.plan["media_plan_id"] = x.fingerprint(m.media_identity(self.plan))
        identity = dict(media_plan_id=self.plan["media_plan_id"],
            official_sources=dict(commit=c.PINNED_COMMIT, files={"fake.py": "c" * 64}),
            dtype="float32", modalities=list(m.MEDIA_KEYS), encoder_hashes={"clip": {"revision": "a" * 40}})
        self.state = dict(identity=identity, run_id=x.fingerprint(identity))
        self.complete = dict(status="MEDIA_6_CPU_VERIFIED", checked=6, training_ready=False,
            batch_indices=[0], rows=[r["clip_id"] for r in m.select_rows(self.plan, "batch", 0, 500)],
            run_id=self.state["run_id"], media_plan_id=self.plan["media_plan_id"])
        for name, value in [("plan.json", self.plan), ("state.json", self.state), ("complete.json", self.complete)]:
            c.atomic_json(self.root / name, value)
        self.review = [{**{k: r[k] for k in m.ROW_KEYS}, "caption_candidate": "Traffic and wind.",
            "caption_final": "Traffic and wind.", "caption_review_status": "approved",
            "content_review_status": "approved", "reviewer": "fixture reviewer",
            "reviewed_at": "2026-10-07T12:00:00+00:00", "acoustic_only_reviewed": "true"}
            for r in self.plan["rows"]]
        self.frozen = dict(status="FROZEN_REVIEWED_CAPTIONS", media_plan_id=self.plan["media_plan_id"], rows=self.review)
        c.atomic_json(self.root / "frozen.json", self.frozen)

    def contract(self):
        return b.media_contract(self.root / "plan.json", self.root / "state.json",
                                self.root / "complete.json", c.sha256_file(self.root / "complete.json"))

    def test_completion_requires_exact_order_and_identity(self):
        self.contract()
        for key, value in [("rows", list(reversed(self.complete["rows"]))), ("run_id", "e" * 64),
                           ("checked", 5), ("training_ready", True), ("batch_indices", [1])]:
            altered = {**self.complete, key: value}
            c.atomic_json(self.root / "complete.json", altered)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.contract()

    def test_missing_caption_or_content_approval_blocks_freeze(self):
        b.validate_review(self.plan, self.review)
        for key, value in [("caption_review_status", "pending"), ("content_review_status", "pending"),
                           ("caption_final", ""), ("reviewer", ""), ("acoustic_only_reviewed", "false"),
                           ("reviewed_at", "2026-10-07T12:00:00"), ("sha256", "e" * 64)]:
            rows = copy.deepcopy(self.review)
            rows[0][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                b.validate_review(self.plan, rows)

    def test_template_preserves_candidates_and_leaves_pending(self):
        candidates = [dict(training_id=r["clip_id"], split=r["split"], caption_candidate="Traffic.")
                      for r in self.plan["rows"]]
        path = self.root / "candidates.csv"
        path.write_bytes(b.csv_bytes(candidates, tuple(candidates[0])))
        rows = b.review_template(self.plan, path)
        self.assertTrue(all(r["caption_candidate"] == "Traffic." and r["caption_final"] == ""
                            and r["content_review_status"] == "pending" for r in rows))
        with self.assertRaises(ValueError):
            b.validate_review(self.plan, rows)

    def test_freeze_preview_has_no_writes(self):
        review = self.root / "review.csv"
        review.write_bytes(b.csv_bytes(self.review, b.REVIEW_FIELDS))
        output = self.root / "not-written.json"
        self.assertEqual(b.main(["freeze", "--plan", str(self.root / "plan.json"), "--review", str(review),
                                 "--output", str(output)]), 0)
        self.assertFalse(output.exists())

    def test_members_reject_traversal_and_symlink_escape(self):
        for value in ("../elsewhere", "/absolute"):
            with self.assertRaises(ValueError):
                b.member(self.root, value)
        (self.root / "escape").symlink_to(self.root.parent)
        with self.assertRaises(ValueError):
            b.member(self.root, "escape/file")

    def text_fixture(self, stack, token_count=4):
        """Real CPU tensors/serialization with a synthetic encoder; no GPU inference."""
        import torch
        calls = []

        class Encoder:
            tokenizer = SimpleNamespace(encode=lambda text: list(range(token_count)))

            def __init__(self, **kwargs):
                self_options = kwargs
                assert self_options["tod_vae_ckpt"] is None

            def eval(self):
                return self

            def cuda(self):
                return self

            def encode_text(self, texts):
                calls.extend(texts)
                return torch.ones(1, 77, 1024)

        weights = {**self.state["identity"]["encoder_hashes"], "synchformer": {"path": "unused", "sha256": "d" * 64}}
        state = copy.deepcopy(self.state)
        state["identity"]["encoder_hashes"] = m.weight_identity(weights)
        state["run_id"] = x.fingerprint(state["identity"])
        official = state["identity"]["official_sources"]
        for target, value in [("fine_tune.official_common.verify_official_repo", official),
                              ("fine_tune.official_common.activate_official_repo", official),
                              ("fine_tune.official_extract.inventory_weights", weights),
                              ("fine_tune.official_extract.runtime_versions", {"fixture": True}),
                              ("torch.cuda.is_available", True), ("torch.cuda.get_device_name", "CPU fixture"),
                              ("fine_tune.official_extract.local_encoder_resolution", nullcontext())]:
            stack.enter_context(patch(target, return_value=value))
        stack.enter_context(patch.dict("sys.modules", {"mmaudio.model.utils.features_utils": SimpleNamespace(FeaturesUtils=Encoder)}))
        args = SimpleNamespace(official_repo=self.root, execute=True, resume=False,
            frozen=self.root / "frozen.json", frozen_sha256=c.sha256_file(self.root / "frozen.json"),
            output=self.root / "text-only")
        return args, state, calls

    def test_text_only_resume_checks_existing_tensors_without_reencoding(self):
        with ExitStack() as stack:
            args, state, calls = self.text_fixture(stack)
            result = b.extract_text(args, self.plan, state, self.frozen)
            self.assertEqual(result["checked"], 6)
            self.assertEqual(len(calls), 6)
            args.resume = True
            b.extract_text(args, self.plan, state, self.frozen)
            self.assertEqual(len(calls), 6)
            data, _ = x.row_paths(args.output / "row_cache/train", 0)
            with data.open("ab") as stream:
                stream.write(b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                b.extract_text(args, self.plan, state, self.frozen)

    def test_token_overflow_stops_before_any_text_forward(self):
        with ExitStack() as stack:
            args, state, calls = self.text_fixture(stack, token_count=76)
            with self.assertRaisesRegex(ValueError, "truncated"):
                b.extract_text(args, self.plan, state, self.frozen)
            self.assertEqual(calls, [])
            self.assertFalse((args.output / "TEXT_COMPLETE.json").exists())

    def caches(self):
        import torch
        media_root = self.root / "media"
        text_root = self.root / "text"
        identity = dict(media_plan_id=self.plan["media_plan_id"], media_run_id=self.state["run_id"],
            frozen_sha256=c.sha256_file(self.root / "frozen.json"),
            encoder_hashes=self.state["identity"]["encoder_hashes"],
            official_sources=self.state["identity"]["official_sources"])
        state = dict(identity=identity, run_id=x.fingerprint(identity))
        c.atomic_json(text_root / "text_state.json", state)
        c.atomic_json(text_root / "TEXT_COMPLETE.json", dict(status="TEXT_CPU_VERIFIED", checked=6,
            run_id=state["run_id"], frozen_sha256=identity["frozen_sha256"], truncations=0, max_tokens=6,
            state_sha256=c.sha256_file(text_root / "text_state.json")))
        for row in self.plan["rows"]:
            data, receipt = x.row_paths(media_root / "row_cache" / row["split"], row["split_row"])
            data.parent.mkdir(parents=True, exist_ok=True)
            torch.save({key: torch.full(x.feature_shapes(self.plan["spec"])[key], row["manifest_row"] + 1.)
                        for key in m.MEDIA_KEYS}, data)
            c.atomic_json(receipt, dict(status="MEDIA_COMPLETE", run_id=self.state["run_id"],
                row_index=row["split_row"], clip_id=row["clip_id"], split=row["split"],
                manifest_row=row["manifest_row"], source_recording_id=row["source_recording_id"],
                source_sha256=row["sha256"], feature_sha256=c.sha256_file(data),
                normalize_audio=c.NORMALIZE_AUDIO[row["split"]], text_features_extracted=False, training_ready=False))
            data, receipt = x.row_paths(text_root / "row_cache" / row["split"], row["split_row"])
            data.parent.mkdir(parents=True, exist_ok=True)
            torch.save(torch.full((77, 1024), row["manifest_row"] + 10.), data)
            c.atomic_json(receipt, dict(status="TEXT_COMPLETE", run_id=state["run_id"],
                row_index=row["split_row"], clip_id=row["clip_id"], split=row["split"],
                label="Traffic and wind.", feature_sha256=c.sha256_file(data)))
        return SimpleNamespace(plan=self.root / "plan.json", state=self.root / "state.json",
            complete=self.root / "complete.json", complete_sha256=c.sha256_file(self.root / "complete.json"),
            frozen=self.root / "frozen.json", frozen_sha256=c.sha256_file(self.root / "frozen.json"),
            media_root=media_root, text_root=text_root, output=self.root / "bundle", execute=True)

    def test_real_cpu_memmap_roundtrip_and_relocation(self):
        import shutil
        import torch
        import tensordict as td
        args = self.caches()
        original_hashes = x.file_hashes(args.media_root)
        result = b.assemble(args, self.plan, self.state, self.frozen)
        self.assertEqual(result["status"], "FEATURE_BUNDLE_READY")
        self.assertEqual(x.file_hashes(args.media_root), original_hashes)
        moved = self.root / "different-machine"
        shutil.copytree(args.output, moved)
        ready = moved / "BUNDLE_READY.json"
        b.verify_bundle(ready, result["sha256"])
        mmap = td.TensorDict.load_memmap(moved / "vgg-val")
        self.assertTrue(torch.equal(mmap["mean"][0], torch.full((345, 40), 3.)))
        self.assertTrue(torch.equal(mmap["text_features"][1], torch.full((77, 1024), 13.)))
        del mmap
        (moved / "vgg-val.tsv").write_text("id\tlabel\nwrong\tWrong.\n")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            b.verify_bundle(ready, result["sha256"])

    def test_changed_media_payload_cannot_be_assembled(self):
        args = self.caches()
        (args.media_root / "row_cache/train/00000000.pth").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "hash"):
            b.assemble(args, self.plan, self.state, self.frozen)
        self.assertFalse((args.output / "BUNDLE_READY.json").exists())

    def test_text_completion_cannot_claim_another_caption_freeze(self):
        args = self.caches()
        with self.assertRaisesRegex(ValueError, "Text completion"):
            b.verify_text_contract(args.text_root, self.plan, self.state, "e" * 64)


if __name__ == "__main__":
    unittest.main()
