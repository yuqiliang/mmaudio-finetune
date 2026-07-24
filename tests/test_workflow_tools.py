from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.apply_source_review import apply_review, read_xlsx_review
from scripts.check_clip_quality import QualityConfig, check_row, rate
from scripts.sync_manifest_to_drive import sync_row


def write_minimal_review_xlsx(path: Path, rows: list[list[str]]) -> None:
    cells = []
    for row_number, values in enumerate(rows, start=1):
        row_cells = []
        for column_number, value in enumerate(values, start=1):
            column = ""
            number = column_number
            while number:
                number, remainder = divmod(number - 1, 26)
                column = chr(ord("A") + remainder) + column
            escaped = (
                value.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )
            row_cells.append(
                f'<c r="{column}{row_number}" t="inlineStr"><is><t>{escaped}</t></is></c>'
            )
        cells.append(f'<row r="{row_number}">{"".join(row_cells)}</row>')

    workbook = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        '<sheets><sheet name="Source Review" sheetId="1" r:id="rId1"/></sheets></workbook>'
    )
    relationships = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
        'Target="worksheets/sheet1.xml"/></Relationships>'
    )
    sheet = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f'<sheetData>{"".join(cells)}</sheetData></worksheet>'
    )
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", workbook)
        archive.writestr("xl/_rels/workbook.xml.rels", relationships)
        archive.writestr("xl/worksheets/sheet1.xml", sheet)


class SourceReviewTests(unittest.TestCase):
    def test_reads_xlsx_and_applies_whole_source_decisions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            workbook = Path(temporary) / "review.xlsx"
            write_minimal_review_xlsx(
                workbook,
                [
                    ["MMAudio Source Video Manual Review"],
                    [
                        "Source File",
                        "Split",
                        "Music",
                        "Decision",
                        "Keep From (s)",
                        "Notes",
                    ],
                    ["old.mov", "train", "No", "Keep", "8", "clean"],
                    ["new.mp4", "test", "Yes", "Exclude whole source", "", "music"],
                ],
            )
            review = read_xlsx_review(workbook, "Source Review")

        sources = [
            {
                "source_key": "old::old.mov",
                "source_dataset_id": "old",
                "source_file": "old.mov",
                "split": "train",
            },
            {
                "source_key": "new::new.mp4",
                "source_dataset_id": "new",
                "source_file": "new.mp4",
                "split": "test",
            },
        ]
        clips = [
            {
                "source_dataset_id": "old",
                "source_file": "old.mov",
                "combined_clip_id": "old::old__000000",
                "clip_id": "old__000000",
                "split": "train",
                "source_start_seconds": "0",
                "absolute_path": "/tmp/old.mp4",
            },
            {
                "source_dataset_id": "old",
                "source_file": "old.mov",
                "combined_clip_id": "old::old__000001",
                "clip_id": "old__000001",
                "split": "train",
                "source_start_seconds": "8",
                "absolute_path": "/tmp/old-1.mp4",
            },
            {
                "source_dataset_id": "new",
                "source_file": "new.mp4",
                "combined_clip_id": "new::new__000000",
                "clip_id": "new__000000",
                "split": "test",
                "absolute_path": "/tmp/new.mp4",
            },
        ]
        kept, excluded, pending, final_clips = apply_review(
            review,
            sources,
            clips,
            allow_pending=False,
        )
        self.assertEqual([row["source_file"] for row in kept], ["old.mov"])
        self.assertEqual([row["source_file"] for row in excluded], ["new.mp4"])
        self.assertEqual(pending, [])
        self.assertEqual(len(final_clips), 1)
        self.assertEqual(final_clips[0]["training_id"], "old__old__000001")

    def test_pending_decision_stops_final_manifest(self) -> None:
        review = [{"Source File": "source.mp4", "Decision": "Not decided"}]
        sources = [
            {
                "source_key": "new::source.mp4",
                "source_dataset_id": "new",
                "source_file": "source.mp4",
                "split": "train",
            }
        ]
        with self.assertRaises(RuntimeError):
            apply_review(review, sources, [], allow_pending=False)


class QualityControlTests(unittest.TestCase):
    def test_missing_file_is_reported_without_mutation(self) -> None:
        row = {
            "training_id": "missing",
            "split": "train",
            "source_file": "source.mp4",
            "absolute_path": "/path/that/does/not/exist.mp4",
        }
        result = check_row(row, QualityConfig(decode_audio=False))
        self.assertEqual(result["qc_status"], "fail")
        self.assertEqual(result["qc_reasons"], "missing_file")

    def test_fractional_frame_rate(self) -> None:
        self.assertAlmostEqual(rate("25/1"), 25.0)
        self.assertAlmostEqual(rate("30000/1001"), 29.97002997)


class DriveSyncTests(unittest.TestCase):
    def test_copy_then_skip_verified_destination(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.mp4"
            source.write_bytes(b"test-video-bytes")
            destination = root / "drive"
            row = {
                "training_id": "dataset__clip_000001",
                "clip_id": "clip_000001",
                "split": "train",
                "absolute_path": str(source),
            }
            copied = sync_row(
                row,
                destination,
                verify="sha256",
                retries=0,
                replace_mismatch=False,
                dry_run=False,
            )
            skipped = sync_row(
                row,
                destination,
                verify="sha256",
                retries=0,
                replace_mismatch=False,
                dry_run=False,
            )
            self.assertEqual(copied["sync_status"], "copied")
            self.assertEqual(skipped["sync_status"], "skipped")
            self.assertEqual(
                Path(copied["absolute_path"]).read_bytes(),
                source.read_bytes(),
            )


if __name__ == "__main__":
    unittest.main()
