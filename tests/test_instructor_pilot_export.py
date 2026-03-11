from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from batch_doc_vqa.datasets.instructor_pilot import (
    Candidate,
    SubmissionRecord,
    select_candidates,
    write_export,
)


def _make_submission(
    *,
    submission_id: str,
    student_db_id: int,
    course_id: int,
    page_count: int,
    canvas_id: str = "",
) -> SubmissionRecord:
    return SubmissionRecord(
        submission_id=submission_id,
        assignment_id=1748,
        assignment_name="Quiz 3",
        course_id=course_id,
        student_db_id=student_db_id,
        student_uni_id="12345678",
        student_full_name=f"Student {student_db_id}",
        attempt=1,
        created="2026-01-01T00:00:00Z",
        updated="2026-01-01T00:00:00Z",
        pdf_relpath=f"submissions/{submission_id}/submission.pdf",
        image_relpaths={
            page: f"{submission_id}/page-{page}.png"
            for page in range(1, page_count + 1)
        },
        canvas_id=canvas_id,
    )


class InstructorPilotExportFilterTests(unittest.TestCase):
    def _touch_media(self, media_root: Path, records: list[SubmissionRecord]) -> None:
        for record in records:
            for relpath in record.image_relpaths.values():
                path = media_root / relpath
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"")

    def test_exact_page_count_filters_out_nonmatching_submissions(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media_root = Path(tmpdir) / "submissions"
            media_root.mkdir(parents=True, exist_ok=True)

            record_4 = _make_submission(
                submission_id="sub-4",
                student_db_id=10,
                course_id=99,
                page_count=4,
            )
            record_5 = _make_submission(
                submission_id="sub-5",
                student_db_id=11,
                course_id=99,
                page_count=5,
            )
            records = [record_4, record_5]
            self._touch_media(media_root, records)

            section_lookup = {
                (10, 99): ["12345"],
                (11, 99): ["12345"],
            }

            candidates, reasons = select_candidates(
                records,
                media_root=media_root,
                required_pages=[1, 2, 3, 4],
                export_pages=[1, 2, 3, 4],
                exact_page_count=4,
                require_canvas_id=False,
                section_lookup=section_lookup,
                require_unique_section=True,
            )

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].submission_id, "sub-4")
            self.assertEqual(reasons["page_count_mismatch"], 1)

    def test_exact_page_count_none_keeps_both_when_other_filters_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media_root = Path(tmpdir) / "submissions"
            media_root.mkdir(parents=True, exist_ok=True)

            record_4 = _make_submission(
                submission_id="sub-4",
                student_db_id=10,
                course_id=99,
                page_count=4,
            )
            record_5 = _make_submission(
                submission_id="sub-5",
                student_db_id=11,
                course_id=99,
                page_count=5,
            )
            records = [record_4, record_5]
            self._touch_media(media_root, records)

            section_lookup = {
                (10, 99): ["12345"],
                (11, 99): ["12345"],
            }

            candidates, reasons = select_candidates(
                records,
                media_root=media_root,
                required_pages=[1, 2, 3, 4],
                export_pages=[1, 2, 3, 4],
                exact_page_count=None,
                require_canvas_id=False,
                section_lookup=section_lookup,
                require_unique_section=True,
            )

            self.assertEqual(len(candidates), 2)
            self.assertEqual(reasons["page_count_mismatch"], 0)

    def test_require_canvas_id_filters_out_missing_canvas_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            media_root = Path(tmpdir) / "submissions"
            media_root.mkdir(parents=True, exist_ok=True)

            record_with_canvas = _make_submission(
                submission_id="sub-with-canvas",
                student_db_id=10,
                course_id=99,
                page_count=4,
                canvas_id="123456",
            )
            record_without_canvas = _make_submission(
                submission_id="sub-no-canvas",
                student_db_id=11,
                course_id=99,
                page_count=4,
                canvas_id="",
            )
            records = [record_with_canvas, record_without_canvas]
            self._touch_media(media_root, records)

            section_lookup = {
                (10, 99): ["12345"],
                (11, 99): ["12345"],
            }

            candidates, reasons = select_candidates(
                records,
                media_root=media_root,
                required_pages=[1, 2, 3, 4],
                export_pages=[1, 2, 3, 4],
                exact_page_count=4,
                require_canvas_id=True,
                section_lookup=section_lookup,
                require_unique_section=True,
            )

            self.assertEqual(len(candidates), 1)
            self.assertEqual(candidates[0].submission_id, "sub-with-canvas")
            self.assertEqual(reasons["missing_canvas_id"], 1)

    def test_write_export_emits_submission_metadata_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            media_root = root / "submissions"
            media_root.mkdir(parents=True, exist_ok=True)

            source_img = media_root / "sub-4" / "page-1.png"
            source_img.parent.mkdir(parents=True, exist_ok=True)
            source_img.write_bytes(b"")

            candidate = Candidate(
                submission_id="sub-4",
                assignment_id=1748,
                assignment_name="Quiz 3",
                course_id=99,
                student_db_id=10,
                student_uni_id="12345678",
                student_full_name="Student 10",
                section_number="12345",
                attempt=1,
                created="2026-01-01T00:00:00Z",
                updated="2026-01-01T00:00:00Z",
                pdf_relpath="submissions/sub-4/submission.pdf",
                pages=[1],
                image_relpaths={1: "sub-4/page-1.png"},
                grade=9.5,
                question_grades="5,4.5",
                graded_at="2026-01-02T00:00:00Z",
                canvas_id="987654321",
                submission_comment_count=3,
                grade_summary_comment_count=1,
            )

            images_dir = root / "export" / "images"
            doc_info_out = root / "export" / "doc_info.csv"
            test_ids_out = root / "export" / "test_ids.csv"
            submission_metadata_out = root / "export" / "submission_metadata.csv"
            manifest_out = root / "export" / "dataset_manifest.json"

            manifest = write_export(
                [candidate],
                media_root=media_root,
                images_dir=images_dir,
                doc_info_out=doc_info_out,
                test_ids_out=test_ids_out,
                submission_metadata_out=submission_metadata_out,
                manifest_out=manifest_out,
                link_mode="copy",
            )

            self.assertTrue(submission_metadata_out.exists())
            with open(submission_metadata_out, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]
            self.assertEqual(len(lines), 2)
            self.assertIn("submission_comment_count", lines[0])
            self.assertIn("grade_summary_comment_count", lines[0])
            self.assertIn("canvas_id", lines[0])
            self.assertIn("sub-4", lines[1])
            self.assertIn(",3,1", lines[1])
            self.assertEqual(
                manifest["artifacts"]["submission_metadata_csv"],
                str(submission_metadata_out),
            )


if __name__ == "__main__":
    unittest.main()
