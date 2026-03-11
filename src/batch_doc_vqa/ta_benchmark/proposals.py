#!/usr/bin/env python3
"""Generate TA annotation proposals from model run outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from .constants import PII_WARNING_BANNER, TA_SCHEMA_VERSION
from .io_utils import (
    build_result_basename_index,
    load_doc_info,
    load_test_ids,
    lookup_result_entry,
    read_json,
    write_json,
)


console = Console()


def _normalize_entry(raw_entry: Any) -> dict[str, Any]:
    if isinstance(raw_entry, list):
        if raw_entry and isinstance(raw_entry[0], dict):
            return dict(raw_entry[0])
        return {}
    if isinstance(raw_entry, dict):
        return dict(raw_entry)
    return {}


def _pick_best_value(page_entries: dict[int, dict[str, Any]], key: str) -> tuple[str, Optional[int]]:
    for page in sorted(page_entries):
        value = str(page_entries[page].get(key, "") or "").strip()
        if value:
            return value, page
    return "", None


def _confidence_hints(page_entries: dict[int, dict[str, Any]]) -> dict[str, Any]:
    name, _ = _pick_best_value(page_entries, "student_full_name")
    uni_id, _ = _pick_best_value(page_entries, "university_id")
    section, _ = _pick_best_value(page_entries, "section_number")
    filled = sum(1 for item in (name, uni_id, section) if item)
    completeness = filled / 3.0

    avg_total_tokens = 0.0
    token_rows = 0
    for entry in page_entries.values():
        token_usage = entry.get("_token_usage")
        if isinstance(token_usage, dict):
            total_tokens = token_usage.get("total_tokens")
            try:
                avg_total_tokens += float(total_tokens)
                token_rows += 1
            except (TypeError, ValueError):
                continue
    if token_rows:
        avg_total_tokens /= float(token_rows)

    return {
        "identity_fields_present": int(filled),
        "identity_completeness": float(completeness),
        "avg_total_tokens": float(avg_total_tokens),
        "missing_name": not bool(name),
        "missing_id": not bool(uni_id),
        "missing_section": not bool(section),
    }


def _coerce_page(value: Any, fallback: int) -> int:
    try:
        page = int(value)
    except (TypeError, ValueError):
        return int(fallback)
    return int(page) if page > 0 else int(fallback)


def _normalize_problem_uid(problem: dict[str, Any], page: int, idx: int) -> str:
    problem_uid = str(problem.get("problem_uid", "") or "").strip()
    if problem_uid:
        return problem_uid

    problem_number = str(problem.get("problem_number", "") or "").strip()
    normalized = "".join(ch if ch.isalnum() else "_" for ch in problem_number).strip("_")
    if normalized:
        return f"p_{normalized.lower()}"
    return f"p_p{int(page)}_{idx + 1}"


def _extract_optional_structured_fields(page_entries: dict[int, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    merged_evidence: dict[str, dict[str, Any]] = {}
    merged_problems: dict[str, dict[str, Any]] = {}
    used_evidence_ids: set[str] = set()

    for page, entry in sorted(page_entries.items()):
        evidence_regions = entry.get("evidence_regions")
        if isinstance(evidence_regions, list):
            for e_idx, region in enumerate(evidence_regions):
                if not isinstance(region, dict):
                    continue
                normalized = dict(region)
                evidence_id = str(normalized.get("evidence_id", "")).strip()
                if not evidence_id:
                    evidence_id = f"ev_p{int(page)}_{e_idx + 1}"
                if evidence_id in used_evidence_ids:
                    evidence_id = f"{evidence_id}_{int(page)}_{e_idx + 1}"
                used_evidence_ids.add(evidence_id)

                normalized["evidence_id"] = evidence_id
                normalized["page"] = _coerce_page(normalized.get("page"), page)
                if "kind" not in normalized or not str(normalized.get("kind", "")).strip():
                    normalized["kind"] = "other"
                merged_evidence[evidence_id] = normalized

        problems = entry.get("problems")
        if isinstance(problems, list):
            for p_idx, problem in enumerate(problems):
                if not isinstance(problem, dict):
                    continue
                normalized_problem = dict(problem)
                problem_uid = _normalize_problem_uid(normalized_problem, page, p_idx)
                normalized_problem["problem_uid"] = problem_uid
                normalized_problem["problem_number"] = str(normalized_problem.get("problem_number", "") or "")
                normalized_problem["description_text"] = str(normalized_problem.get("description_text", "") or "")
                for key in ("description_evidence_ids", "figure_evidence_ids"):
                    refs = normalized_problem.get(key, [])
                    if not isinstance(refs, list):
                        normalized_problem[key] = []
                        continue
                    cleaned_refs: list[str] = []
                    for ref in refs:
                        text = str(ref or "").strip()
                        if text:
                            cleaned_refs.append(text)
                    normalized_problem[key] = cleaned_refs
                merged_problems[problem_uid] = normalized_problem
    return list(merged_evidence.values()), list(merged_problems.values())


def build_proposals(
    *,
    results_json: str | Path,
    doc_info_csv: str | Path,
    output_path: str | Path,
    test_ids_csv: Optional[str | Path] = None,
    images_dir: Optional[str] = None,
) -> dict[str, Any]:
    results = read_json(results_json)
    if not isinstance(results, dict):
        raise ValueError("results.json must be an object keyed by image path.")
    doc_info = load_doc_info(doc_info_csv)
    gt = load_test_ids(test_ids_csv) if test_ids_csv else {}
    basename_index = build_result_basename_index(results)

    docs_payload: list[dict[str, Any]] = []
    for doc_id in sorted(doc_info):
        page_rows = doc_info[doc_id]
        page_entries: dict[int, dict[str, Any]] = {}
        for row in page_rows:
            filename = str(row.get("filename", "")).strip()
            if not filename:
                continue
            raw_entry = lookup_result_entry(
                results=results,
                basename_index=basename_index,
                filename=filename,
                images_dir=images_dir,
            )
            entry = _normalize_entry(raw_entry)
            if entry:
                page_entries[int(row["page"])] = entry

        name, name_page = _pick_best_value(page_entries, "student_full_name")
        uni_id, id_page = _pick_best_value(page_entries, "university_id")
        section, section_page = _pick_best_value(page_entries, "section_number")
        evidence_regions, problems = _extract_optional_structured_fields(page_entries)
        confidence = _confidence_hints(page_entries)

        gt_row = gt.get(doc_id, {})
        submission_id = str(gt_row.get("submission_id", "") or "").strip() or f"unknown-{doc_id}"
        assignment_id_raw = str(gt_row.get("assignment_id", "") or "").strip()
        try:
            assignment_id = int(assignment_id_raw)
        except ValueError:
            assignment_id = 1
        assignment_id = max(1, assignment_id)

        docs_payload.append(
            {
                "doc_id": doc_id,
                "schema_version": TA_SCHEMA_VERSION,
                "submission_id": submission_id,
                "assignment_id": assignment_id,
                "template_version_id": "",
                "evidence_regions": evidence_regions,
                "problems": problems,
                "tier2": {
                    "rubric_scores": [],
                    "feedback": [],
                },
                "review": {
                    "annotator_id": "",
                    "status": "draft",
                    "updated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "proposal_action": "",
                },
                "proposal": {
                    "identity_guess": {
                        "student_full_name": name,
                        "university_id": uni_id,
                        "section_number": section,
                        "name_source_page": name_page,
                        "id_source_page": id_page,
                        "section_source_page": section_page,
                    },
                    "confidence_hints": confidence,
                    "source_pages": sorted(page_entries.keys()),
                    "proposal_version": "ta_proposals_v1",
                },
            }
        )

    payload = {
        "proposal_version": "ta_proposals_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "results_json": str(Path(results_json).expanduser().resolve(strict=False)),
        "doc_info_csv": str(Path(doc_info_csv).expanduser().resolve(strict=False)),
        "test_ids_csv": str(Path(test_ids_csv).expanduser().resolve(strict=False)) if test_ids_csv else None,
        "docs": docs_payload,
    }
    write_json(output_path, payload)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TA label proposals from run outputs.")
    parser.add_argument("--results-json", required=True, help="Path to run results.json")
    parser.add_argument("--doc-info", required=True, help="Path to doc_info.csv")
    parser.add_argument("--output", required=True, help="Output proposal JSON path")
    parser.add_argument("--test-ids", help="Optional test_ids.csv for assignment/submission metadata")
    parser.add_argument("--images-dir", help="Optional images directory for filename resolution")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    console.print(f"[yellow]{PII_WARNING_BANNER}[/yellow]")

    payload = build_proposals(
        results_json=args.results_json,
        doc_info_csv=args.doc_info,
        output_path=args.output,
        test_ids_csv=args.test_ids,
        images_dir=args.images_dir,
    )

    docs = payload.get("docs", [])
    table = Table(title="TA Proposal Generation")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Output", str(Path(args.output).expanduser().resolve(strict=False)))
    table.add_row("Docs", str(len(docs)))
    table.add_row("Proposal version", str(payload.get("proposal_version", "")))
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
