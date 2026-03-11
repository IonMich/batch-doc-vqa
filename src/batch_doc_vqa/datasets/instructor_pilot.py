#!/usr/bin/env python3
"""Profile and export Instructor Pilot submissions into benchmark-compatible format."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sqlite3
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.table import Table

from ..ta_benchmark.constants import PII_WARNING_BANNER


console = Console()


@dataclass
class SubmissionRecord:
    submission_id: str
    assignment_id: int
    assignment_name: str
    course_id: int
    student_db_id: Optional[int]
    student_uni_id: str
    student_full_name: str
    attempt: Optional[int]
    created: str
    updated: str
    pdf_relpath: str
    image_relpaths: Dict[int, str] = field(default_factory=dict)
    grade: Optional[float] = None
    question_grades: str = ""
    graded_at: str = ""
    canvas_id: str = ""
    submission_comment_count: int = 0
    grade_summary_comment_count: int = 0


@dataclass
class Candidate:
    submission_id: str
    assignment_id: int
    assignment_name: str
    course_id: int
    student_db_id: int
    student_uni_id: str
    student_full_name: str
    section_number: str
    attempt: Optional[int]
    created: str
    updated: str
    pdf_relpath: str
    pages: List[int]
    image_relpaths: Dict[int, str]
    grade: Optional[float] = None
    question_grades: str = ""
    graded_at: str = ""
    canvas_id: str = ""
    submission_comment_count: int = 0
    grade_summary_comment_count: int = 0


def parse_int_csv(raw: Optional[str], *, flag_name: str) -> List[int]:
    if raw is None:
        return []
    values: List[int] = []
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        try:
            values.append(int(text))
        except ValueError as exc:
            raise ValueError(f"{flag_name} must be comma-separated integers; got {raw!r}") from exc
    return sorted(set(values))


def normalize_uni_id(value: object) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.isdigit():
        return text.zfill(8)
    digits_only = "".join(ch for ch in text if ch.isdigit())
    if digits_only:
        return digits_only[-8:].zfill(8)
    return ""


def normalize_name(first_name: object, last_name: object) -> str:
    first = str(first_name or "").strip()
    last = str(last_name or "").strip()
    full = f"{first} {last}".strip()
    return " ".join(full.split())


def load_section_lookup(conn: sqlite3.Connection) -> Dict[Tuple[int, int], List[str]]:
    query = """
    SELECT
        sss.student_id,
        sec.course_id,
        sec.class_number
    FROM students_student_sections sss
    JOIN sections_section sec ON sec.id = sss.section_id
    """
    grouped: Dict[Tuple[int, int], set[str]] = defaultdict(set)
    for student_id, course_id, class_number in conn.execute(query):
        if student_id is None or course_id is None or class_number is None:
            continue
        grouped[(int(student_id), int(course_id))].add(str(class_number).strip())
    return {key: sorted(values) for key, values in grouped.items() if values}


def load_submissions(
    conn: sqlite3.Connection,
    *,
    assignment_ids: Optional[Sequence[int]] = None,
) -> Dict[str, SubmissionRecord]:
    where = ""
    params: List[object] = []
    if assignment_ids:
        placeholders = ",".join("?" for _ in assignment_ids)
        where = f"WHERE ps.assignment_id IN ({placeholders})"
        params.extend(list(assignment_ids))

    query = f"""
    SELECT
        ps.id AS submission_id,
        ps.assignment_id,
        a.name AS assignment_name,
        a.course_id,
        ps.student_id,
        st.uni_id,
        st.first_name,
        st.last_name,
        ps.attempt,
        COALESCE(ps.created, '') AS created,
        COALESCE(ps.updated, '') AS updated,
        ps.grade,
        COALESCE(ps.question_grades, '') AS question_grades,
        COALESCE(ps.graded_at, '') AS graded_at,
        COALESCE(ps.canvas_id, '') AS canvas_id,
        COALESCE(ps.pdf, '') AS pdf_relpath,
        psi.page,
        psi.image
    FROM submissions_papersubmission ps
    JOIN assignments_assignment a ON a.id = ps.assignment_id
    LEFT JOIN students_student st ON st.id = ps.student_id
    LEFT JOIN submissions_papersubmissionimage psi ON psi.submission_id = ps.id
    {where}
    ORDER BY ps.assignment_id, ps.id, psi.page
    """

    submissions: Dict[str, SubmissionRecord] = {}
    for row in conn.execute(query, params):
        (
            submission_id,
            assignment_id,
            assignment_name,
            course_id,
            student_db_id,
            uni_id,
            first_name,
            last_name,
            attempt,
            created,
            updated,
            grade,
            question_grades,
            graded_at,
            canvas_id,
            pdf_relpath,
            page,
            image_relpath,
        ) = row

        key = str(submission_id)
        record = submissions.get(key)
        if record is None:
            record = SubmissionRecord(
                submission_id=key,
                assignment_id=int(assignment_id),
                assignment_name=str(assignment_name or ""),
                course_id=int(course_id),
                student_db_id=int(student_db_id) if student_db_id is not None else None,
                student_uni_id=normalize_uni_id(uni_id),
                student_full_name=normalize_name(first_name, last_name),
                attempt=int(attempt) if attempt is not None else None,
                created=str(created or ""),
                updated=str(updated or ""),
                grade=float(grade) if grade is not None else None,
                question_grades=str(question_grades or ""),
                graded_at=str(graded_at or ""),
                canvas_id=str(canvas_id or ""),
                pdf_relpath=str(pdf_relpath or ""),
                image_relpaths={},
            )
            submissions[key] = record

        if page is None or image_relpath is None:
            continue
        record.image_relpaths[int(page)] = str(image_relpath)

    return submissions


def load_submission_comment_counts(
    conn: sqlite3.Connection,
) -> Dict[str, Tuple[int, int]]:
    query = """
    SELECT
        COALESCE(paper_submission_id, '') AS submission_id,
        COUNT(*) AS total_comments,
        SUM(CASE WHEN is_grade_summary = 1 THEN 1 ELSE 0 END) AS grade_summary_comments
    FROM submissions_submissioncomment
    GROUP BY paper_submission_id
    """
    counts: Dict[str, Tuple[int, int]] = {}
    for submission_id, total_comments, grade_summary_comments in conn.execute(query):
        key = str(submission_id or "").strip()
        if not key:
            continue
        counts[key] = (
            int(total_comments or 0),
            int(grade_summary_comments or 0),
        )
    return counts


def profile_assignments(submissions: Iterable[SubmissionRecord], required_pages: Sequence[int]) -> List[dict]:
    required = set(required_pages)
    rows: Dict[int, dict] = {}
    unique_students: Dict[int, set[int]] = defaultdict(set)

    for record in submissions:
        row = rows.setdefault(
            record.assignment_id,
            {
                "assignment_id": record.assignment_id,
                "assignment_name": record.assignment_name,
                "submissions": 0,
                "linked_students": 0,
                "with_required_pages": 0,
            },
        )
        row["submissions"] += 1

        if record.student_db_id is not None and record.student_uni_id:
            row["linked_students"] += 1
            unique_students[record.assignment_id].add(record.student_db_id)

        if required.issubset(set(record.image_relpaths)):
            row["with_required_pages"] += 1

    profiled: List[dict] = []
    for assignment_id, row in rows.items():
        row["unique_students"] = len(unique_students.get(assignment_id, set()))
        row["has_duplicates"] = row["linked_students"] > row["unique_students"]
        profiled.append(row)

    profiled.sort(key=lambda item: item["submissions"], reverse=True)
    return profiled


def extract_image_suffix(submission_id: str, image_relpath: str) -> str:
    stem = Path(image_relpath).stem
    token = stem.split("-")[-1].upper()
    if re.fullmatch(r"[A-Z0-9]+", token):
        return token
    fallback = "".join(ch for ch in submission_id.upper() if ch.isalnum())
    return (fallback[:8] or "UNKNOWN").upper()


def select_candidates(
    submissions: Iterable[SubmissionRecord],
    *,
    media_root: Path,
    required_pages: Sequence[int],
    export_pages: Optional[Sequence[int]],
    exact_page_count: Optional[int],
    require_canvas_id: bool,
    section_lookup: Dict[Tuple[int, int], List[str]],
    require_unique_section: bool,
) -> Tuple[List[Candidate], Counter]:
    required = set(required_pages)
    export_filter = set(export_pages) if export_pages else None
    reasons: Counter = Counter()
    candidates: List[Candidate] = []

    for record in submissions:
        if record.student_db_id is None:
            reasons["missing_student_id"] += 1
            continue
        if not record.student_uni_id:
            reasons["missing_uni_id"] += 1
            continue
        if not record.student_full_name:
            reasons["missing_student_name"] += 1
            continue
        if require_canvas_id and not str(record.canvas_id).strip():
            reasons["missing_canvas_id"] += 1
            continue
        if exact_page_count is not None and len(record.image_relpaths) != exact_page_count:
            reasons["page_count_mismatch"] += 1
            continue
        if not required.issubset(set(record.image_relpaths)):
            reasons["missing_required_pages"] += 1
            continue

        section_candidates = section_lookup.get((record.student_db_id, record.course_id), [])
        if require_unique_section and len(section_candidates) != 1:
            reasons["non_unique_or_missing_section"] += 1
            continue
        section_number = section_candidates[0] if len(section_candidates) == 1 else ""

        selected_pages = sorted(record.image_relpaths) if export_filter is None else sorted(
            page for page in record.image_relpaths if page in export_filter
        )
        if not selected_pages:
            reasons["no_pages_selected_for_export"] += 1
            continue

        missing_files = False
        for page in selected_pages:
            image_path = resolve_media_file(media_root, record.image_relpaths[page])
            if not image_path.exists():
                missing_files = True
                break
        if missing_files:
            reasons["missing_image_files"] += 1
            continue

        candidates.append(
            Candidate(
                submission_id=record.submission_id,
                assignment_id=record.assignment_id,
                assignment_name=record.assignment_name,
                course_id=record.course_id,
                student_db_id=record.student_db_id,
                student_uni_id=record.student_uni_id,
                student_full_name=record.student_full_name,
                section_number=section_number,
                attempt=record.attempt,
                created=record.created,
                updated=record.updated,
                pdf_relpath=record.pdf_relpath,
                pages=selected_pages,
                image_relpaths=dict(record.image_relpaths),
                grade=record.grade,
                question_grades=record.question_grades,
                graded_at=record.graded_at,
                canvas_id=record.canvas_id,
                submission_comment_count=record.submission_comment_count,
                grade_summary_comment_count=record.grade_summary_comment_count,
            )
        )

    return candidates, reasons


def dedupe_candidates(candidates: Iterable[Candidate], *, policy: str) -> Tuple[List[Candidate], int]:
    grouped: Dict[Tuple[int, int], List[Candidate]] = defaultdict(list)
    for candidate in candidates:
        grouped[(candidate.assignment_id, candidate.student_db_id)].append(candidate)

    def sort_key(item: Candidate) -> Tuple[str, str, int, str]:
        return (
            item.updated or "",
            item.created or "",
            int(item.attempt or 0),
            item.submission_id,
        )

    deduped: List[Candidate] = []
    discarded = 0
    for _key, group in grouped.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue
        ordered = sorted(group, key=sort_key)
        chosen = ordered[-1] if policy == "latest" else ordered[0]
        deduped.append(chosen)
        discarded += len(group) - 1

    deduped.sort(key=lambda item: (item.assignment_id, item.student_uni_id, item.submission_id))
    return deduped, discarded


def filter_assignments_by_min_docs(
    candidates: Iterable[Candidate],
    *,
    min_docs_per_assignment: int,
) -> Tuple[List[Candidate], Dict[int, int]]:
    per_assignment = Counter(candidate.assignment_id for candidate in candidates)
    filtered = [
        candidate
        for candidate in candidates
        if per_assignment[candidate.assignment_id] >= min_docs_per_assignment
    ]
    return filtered, dict(per_assignment)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prepare_path_for_write(path: Path, *, force: bool) -> None:
    if path.exists():
        if not force:
            raise FileExistsError(f"Path already exists: {path}")
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()


def materialize_file(src: Path, dst: Path, *, mode: str) -> None:
    ensure_parent(dst)
    if dst.exists():
        dst.unlink()
    if mode == "copy":
        shutil.copy2(src, dst)
        return
    if mode == "hardlink":
        os.link(src, dst)
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return
    raise ValueError(f"Unknown link mode: {mode}")


def resolve_media_file(media_root: Path, relative_path: str) -> Path:
    rel = Path(relative_path)
    candidates: List[Path] = [media_root / rel]

    if rel.parts and rel.parts[0] == "submissions":
        candidates.append(media_root.parent / rel)
    else:
        candidates.append(media_root / "submissions" / rel)
        candidates.append(media_root.parent / "submissions" / rel)

    seen: set[Path] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    return candidates[0]


def write_export(
    candidates: Sequence[Candidate],
    *,
    media_root: Path,
    images_dir: Path,
    doc_info_out: Path,
    test_ids_out: Path,
    submission_metadata_out: Path,
    manifest_out: Path,
    link_mode: str,
) -> Dict[str, object]:
    images_dir.mkdir(parents=True, exist_ok=True)
    ensure_parent(doc_info_out)
    ensure_parent(test_ids_out)
    ensure_parent(submission_metadata_out)
    ensure_parent(manifest_out)

    doc_rows: List[Dict[str, object]] = []
    ids_rows: List[Dict[str, object]] = []
    metadata_rows: List[Dict[str, object]] = []

    for doc_idx, candidate in enumerate(candidates):
        suffix_source_page = candidate.pages[0]
        suffix = extract_image_suffix(
            candidate.submission_id,
            candidate.image_relpaths[suffix_source_page],
        )

        ids_rows.append(
            {
                "doc": doc_idx,
                "student_id": candidate.student_uni_id,
                "student_full_name": candidate.student_full_name,
                "section_number": candidate.section_number,
                "assignment_id": candidate.assignment_id,
                "submission_id": candidate.submission_id,
            }
        )
        metadata_rows.append(
            {
                "doc": doc_idx,
                "submission_id": candidate.submission_id,
                "canvas_id": candidate.canvas_id,
                "assignment_id": candidate.assignment_id,
                "grade": candidate.grade if candidate.grade is not None else "",
                "question_grades": candidate.question_grades,
                "graded_at": candidate.graded_at,
                "submission_comment_count": candidate.submission_comment_count,
                "grade_summary_comment_count": candidate.grade_summary_comment_count,
            }
        )

        for page in candidate.pages:
            source_path = resolve_media_file(media_root, candidate.image_relpaths[page])
            output_name = f"doc-{doc_idx}-page-{page}-{suffix}.png"
            output_path = images_dir / output_name
            materialize_file(source_path, output_path, mode=link_mode)
            doc_rows.append(
                {
                    "doc": doc_idx,
                    "page": page,
                    "filename": output_name,
                }
            )

    doc_rows.sort(key=lambda item: (int(item["doc"]), int(item["page"])))
    ids_rows.sort(key=lambda item: int(item["doc"]))
    metadata_rows.sort(key=lambda item: int(item["doc"]))

    with open(doc_info_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["doc", "page", "filename"])
        writer.writeheader()
        writer.writerows(doc_rows)

    with open(test_ids_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc",
                "student_id",
                "student_full_name",
                "section_number",
                "assignment_id",
                "submission_id",
            ],
        )
        writer.writeheader()
        writer.writerows(ids_rows)

    with open(submission_metadata_out, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "doc",
                "submission_id",
                "canvas_id",
                "assignment_id",
                "grade",
                "question_grades",
                "graded_at",
                "submission_comment_count",
                "grade_summary_comment_count",
            ],
        )
        writer.writeheader()
        writer.writerows(metadata_rows)

    assignment_counts: Counter = Counter(candidate.assignment_id for candidate in candidates)
    assignment_names = {candidate.assignment_id: candidate.assignment_name for candidate in candidates}
    assignment_rows = [
        {
            "assignment_id": assignment_id,
            "assignment_name": assignment_names.get(assignment_id, ""),
            "documents": count,
        }
        for assignment_id, count in sorted(assignment_counts.items())
    ]

    manifest = {
        "manifest_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "documents": len(candidates),
        "images": len(doc_rows),
        "artifacts": {
            "images_dir": str(images_dir),
            "doc_info_csv": str(doc_info_out),
            "test_ids_csv": str(test_ids_out),
            "submission_metadata_csv": str(submission_metadata_out),
        },
        "assignments": assignment_rows,
    }
    with open(manifest_out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest


def render_profile(rows: Sequence[dict], *, top: int) -> None:
    table = Table(title="Instructor Pilot Assignment Profile")
    table.add_column("Assignment", style="cyan")
    table.add_column("Name", style="white")
    table.add_column("Submissions", justify="right")
    table.add_column("Linked", justify="right")
    table.add_column("Unique Students", justify="right")
    table.add_column("Has Required Pages", justify="right")
    table.add_column("Duplicates", justify="right")

    for row in rows[:top]:
        table.add_row(
            str(row["assignment_id"]),
            str(row["assignment_name"]),
            str(row["submissions"]),
            str(row["linked_students"]),
            str(row["unique_students"]),
            str(row["with_required_pages"]),
            "yes" if row["has_duplicates"] else "no",
        )
    console.print(table)


def render_export_summary(
    *,
    total_submissions: int,
    candidates_pre_dedupe: int,
    discarded_by_reason: Counter,
    deduped_count: int,
    deduped_discarded: int,
    filtered_count: int,
    per_assignment_before_min: Dict[int, int],
    assignment_names: Dict[int, str],
    dry_run: bool,
) -> None:
    summary = Table(show_header=False, box=None)
    summary.add_row("[cyan]Submissions scanned:[/cyan]", str(total_submissions))
    summary.add_row("[cyan]Eligible candidates:[/cyan]", str(candidates_pre_dedupe))
    summary.add_row("[cyan]After dedupe:[/cyan]", f"{deduped_count} (discarded {deduped_discarded})")
    summary.add_row("[cyan]After assignment min-doc filter:[/cyan]", str(filtered_count))
    summary.add_row("[cyan]Mode:[/cyan]", "dry-run" if dry_run else "apply")
    console.print(summary)

    if discarded_by_reason:
        reasons_table = Table(title="Discarded During Candidate Selection")
        reasons_table.add_column("Reason", style="yellow")
        reasons_table.add_column("Count", style="white", justify="right")
        for reason, count in discarded_by_reason.most_common():
            reasons_table.add_row(reason, str(count))
        console.print(reasons_table)

    assignments_table = Table(title="Assignment Counts Before Min-Docs Filter")
    assignments_table.add_column("Assignment", style="cyan")
    assignments_table.add_column("Name", style="white")
    assignments_table.add_column("Docs", style="white", justify="right")
    for assignment_id, count in sorted(per_assignment_before_min.items(), key=lambda item: (-item[1], item[0])):
        assignments_table.add_row(
            str(assignment_id),
            assignment_names.get(assignment_id, ""),
            str(count),
        )
    console.print(assignments_table)


def run_profile(args: argparse.Namespace) -> int:
    console.print(f"[yellow]{PII_WARNING_BANNER}[/yellow]")
    required_pages = parse_int_csv(args.required_pages, flag_name="--required-pages")
    if not required_pages:
        console.print("[red]--required-pages cannot be empty[/red]")
        return 2
    assignment_ids = parse_int_csv(args.assignment_ids, flag_name="--assignment-ids")

    conn = sqlite3.connect(args.db)
    try:
        submissions = load_submissions(
            conn,
            assignment_ids=assignment_ids or None,
        )
    finally:
        conn.close()

    rows = profile_assignments(submissions.values(), required_pages=required_pages)
    if not rows:
        console.print("[yellow]No assignments matched the filters.[/yellow]")
        return 0

    render_profile(rows, top=max(1, args.top))
    return 0


def run_export(args: argparse.Namespace) -> int:
    console.print(f"[yellow]{PII_WARNING_BANNER}[/yellow]")
    required_pages = parse_int_csv(args.required_pages, flag_name="--required-pages")
    if not required_pages:
        console.print("[red]--required-pages cannot be empty[/red]")
        return 2
    export_pages = parse_int_csv(args.export_pages, flag_name="--export-pages") if args.export_pages else []
    exact_page_count: Optional[int] = args.exact_page_count
    if exact_page_count is not None and exact_page_count <= 0:
        console.print("[red]--exact-page-count must be a positive integer[/red]")
        return 2
    assignment_ids = parse_int_csv(args.assignment_ids, flag_name="--assignment-ids")

    db_path = Path(args.db).expanduser().resolve()
    media_root = Path(args.media_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    images_dir = Path(args.images_out).expanduser().resolve() if args.images_out else output_dir / "images"
    doc_info_out = Path(args.doc_info_out).expanduser().resolve() if args.doc_info_out else output_dir / "doc_info.csv"
    test_ids_out = Path(args.test_ids_out).expanduser().resolve() if args.test_ids_out else output_dir / "test_ids.csv"
    submission_metadata_out = (
        Path(args.submission_metadata_out).expanduser().resolve()
        if args.submission_metadata_out
        else output_dir / "submission_metadata.csv"
    )
    manifest_out = Path(args.manifest_out).expanduser().resolve() if args.manifest_out else output_dir / "dataset_manifest.json"

    if not db_path.exists():
        console.print(f"[red]DB path does not exist: {db_path}[/red]")
        return 2
    if not media_root.exists():
        console.print(f"[red]Media root does not exist: {media_root}[/red]")
        return 2

    conn = sqlite3.connect(str(db_path))
    try:
        section_lookup = load_section_lookup(conn)
        comment_counts = load_submission_comment_counts(conn)
        submissions = load_submissions(
            conn,
            assignment_ids=assignment_ids or None,
        )
    finally:
        conn.close()

    for submission_id, record in submissions.items():
        total_comments, summary_comments = comment_counts.get(submission_id, (0, 0))
        record.submission_comment_count = int(total_comments)
        record.grade_summary_comment_count = int(summary_comments)

    candidates, discarded_by_reason = select_candidates(
        submissions.values(),
        media_root=media_root,
        required_pages=required_pages,
        export_pages=export_pages or None,
        exact_page_count=exact_page_count,
        require_canvas_id=bool(args.require_canvas_id),
        section_lookup=section_lookup,
        require_unique_section=(not args.allow_ambiguous_section),
    )
    deduped, deduped_discarded = dedupe_candidates(candidates, policy=args.dedupe_policy)
    filtered, per_assignment_before_min = filter_assignments_by_min_docs(
        deduped,
        min_docs_per_assignment=max(1, int(args.min_docs_per_assignment)),
    )

    assignment_names = {item.assignment_id: item.assignment_name for item in deduped}
    render_export_summary(
        total_submissions=len(submissions),
        candidates_pre_dedupe=len(candidates),
        discarded_by_reason=discarded_by_reason,
        deduped_count=len(deduped),
        deduped_discarded=deduped_discarded,
        filtered_count=len(filtered),
        per_assignment_before_min=per_assignment_before_min,
        assignment_names=assignment_names,
        dry_run=(not args.apply),
    )

    if not filtered:
        console.print("[yellow]No records left after filtering. Nothing to export.[/yellow]")
        return 0

    if not args.apply:
        console.print("[cyan]Dry-run only. Re-run with --apply to write files.[/cyan]")
        return 0

    targets = [images_dir, doc_info_out, test_ids_out, submission_metadata_out, manifest_out]
    for path in targets:
        if path.exists():
            if not args.force:
                console.print(f"[red]Target exists: {path}[/red]")
                console.print("[yellow]Re-run with --force to overwrite targets.[/yellow]")
                return 2

    prepare_path_for_write(images_dir, force=args.force)
    prepare_path_for_write(doc_info_out, force=args.force)
    prepare_path_for_write(test_ids_out, force=args.force)
    prepare_path_for_write(submission_metadata_out, force=args.force)
    prepare_path_for_write(manifest_out, force=args.force)

    manifest = write_export(
        filtered,
        media_root=media_root,
        images_dir=images_dir,
        doc_info_out=doc_info_out,
        test_ids_out=test_ids_out,
        submission_metadata_out=submission_metadata_out,
        manifest_out=manifest_out,
        link_mode=args.link_mode,
    )
    console.print("[green]✅ Export complete[/green]")
    console.print(f"[cyan]Images:[/cyan] {images_dir}")
    console.print(f"[cyan]Doc info:[/cyan] {doc_info_out}")
    console.print(f"[cyan]Test IDs:[/cyan] {test_ids_out}")
    console.print(f"[cyan]Submission metadata:[/cyan] {submission_metadata_out}")
    console.print(f"[cyan]Manifest:[/cyan] {manifest_out}")
    console.print(f"[cyan]Docs:[/cyan] {manifest.get('documents', 0)}")
    console.print(f"[cyan]Images exported:[/cyan] {manifest.get('images', 0)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Utilities for Instructor Pilot -> batch-doc-vqa dataset preparation."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    profile = sub.add_parser("profile", help="Profile assignments and linkage quality")
    profile.add_argument("--db", required=True, help="Path to instructor_pilot db.sqlite3")
    profile.add_argument("--assignment-ids", help="Optional comma-separated assignment IDs")
    profile.add_argument("--required-pages", default="1,3", help="Required pages (comma-separated)")
    profile.add_argument("--top", type=int, default=25, help="Rows to display (default: 25)")
    profile.set_defaults(handler=run_profile)

    export = sub.add_parser("export", help="Export benchmark-compatible dataset artifacts")
    export.add_argument("--db", required=True, help="Path to instructor_pilot db.sqlite3")
    export.add_argument("--media-root", required=True, help="Path to media/submissions root directory")
    export.add_argument("--output-dir", required=True, help="Base output directory")
    export.add_argument("--images-out", help="Override output images directory (default: <output-dir>/images)")
    export.add_argument("--doc-info-out", help="Override doc_info.csv path (default: <output-dir>/doc_info.csv)")
    export.add_argument("--test-ids-out", help="Override test_ids.csv path (default: <output-dir>/test_ids.csv)")
    export.add_argument(
        "--submission-metadata-out",
        help="Override submission_metadata.csv path (default: <output-dir>/submission_metadata.csv)",
    )
    export.add_argument(
        "--manifest-out",
        help="Override manifest path (default: <output-dir>/dataset_manifest.json)",
    )
    export.add_argument("--assignment-ids", help="Optional comma-separated assignment IDs")
    export.add_argument("--required-pages", default="1,3", help="Required pages (comma-separated)")
    export.add_argument(
        "--export-pages",
        help="Optional page whitelist to export (comma-separated). Default: all available pages.",
    )
    export.add_argument(
        "--exact-page-count",
        type=int,
        help="Require submissions to have exactly this many unique pages before export.",
    )
    export.add_argument(
        "--require-canvas-id",
        action="store_true",
        help="Keep only submissions with non-empty canvas_id.",
    )
    export.add_argument(
        "--dedupe-policy",
        choices=["latest", "earliest"],
        default="latest",
        help="How to pick one submission per (assignment, student)",
    )
    export.add_argument(
        "--min-docs-per-assignment",
        type=int,
        default=10,
        help="Drop assignments with fewer docs after dedupe (default: 10)",
    )
    export.add_argument(
        "--allow-ambiguous-section",
        action="store_true",
        help="Allow missing/ambiguous section mapping (section_number exported as empty)",
    )
    export.add_argument(
        "--link-mode",
        choices=["symlink", "copy", "hardlink"],
        default="symlink",
        help="How to materialize image files (default: symlink)",
    )
    export.add_argument(
        "--apply",
        action="store_true",
        help="Write files. Without this flag, command is dry-run only.",
    )
    export.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output targets when --apply is used.",
    )
    export.set_defaults(handler=run_export)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.handler(args))
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
