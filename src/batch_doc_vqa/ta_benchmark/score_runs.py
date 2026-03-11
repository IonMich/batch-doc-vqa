#!/usr/bin/env python3
"""CLI: score TA benchmark predictions against labeled documents."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from .constants import PII_WARNING_BANNER
from .io_utils import (
    build_result_basename_index,
    load_doc_info,
    lookup_result_entry,
    read_json,
    write_json,
)
from .metrics import score_documents
from .schema import iter_label_files


console = Console()


def _normalize_entry(raw_entry: Any) -> dict[str, Any]:
    if isinstance(raw_entry, list):
        if raw_entry and isinstance(raw_entry[0], dict):
            return dict(raw_entry[0])
        return {}
    if isinstance(raw_entry, dict):
        return dict(raw_entry)
    return {}


def _merge_prediction_doc_rows(rows: list[dict[str, Any]], *, doc_id: int) -> dict[str, Any]:
    merged_evidence: dict[str, dict[str, Any]] = {}
    merged_problems: dict[str, dict[str, Any]] = {}
    merged_rubric: dict[tuple[str, str], dict[str, Any]] = {}
    merged_feedback: dict[str, dict[str, Any]] = {}
    template_version_id = ""

    for entry in rows:
        template = str(entry.get("template_version_id", "") or "").strip()
        if template:
            template_version_id = template

        regions = entry.get("evidence_regions", [])
        if isinstance(regions, list):
            for region in regions:
                if not isinstance(region, dict):
                    continue
                evidence_id = str(region.get("evidence_id", "") or "").strip()
                if evidence_id:
                    merged_evidence[evidence_id] = dict(region)

        problems = entry.get("problems", [])
        if isinstance(problems, list):
            for problem in problems:
                if not isinstance(problem, dict):
                    continue
                uid = str(problem.get("problem_uid", "") or "").strip()
                if uid:
                    merged_problems[uid] = dict(problem)

        tier2 = entry.get("tier2", {})
        if isinstance(tier2, dict):
            rubric = tier2.get("rubric_scores", [])
            if isinstance(rubric, list):
                for row in rubric:
                    if not isinstance(row, dict):
                        continue
                    uid = str(row.get("problem_uid", "") or "").strip()
                    criterion = str(row.get("criterion_id", "") or "").strip()
                    if uid and criterion:
                        merged_rubric[(uid, criterion)] = dict(row)
            feedback = tier2.get("feedback", [])
            if isinstance(feedback, list):
                for row in feedback:
                    if not isinstance(row, dict):
                        continue
                    uid = str(row.get("problem_uid", "") or "").strip()
                    if uid:
                        merged_feedback[uid] = dict(row)

    return {
        "doc_id": doc_id,
        "template_version_id": template_version_id,
        "evidence_regions": list(merged_evidence.values()),
        "problems": list(merged_problems.values()),
        "tier2": {
            "rubric_scores": list(merged_rubric.values()),
            "feedback": list(merged_feedback.values()),
        },
    }


def _normalize_predictions_payload(
    payload: Any,
    *,
    doc_info_path: Optional[str],
    images_dir: Optional[str],
) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("docs"), list):
        return [row for row in payload["docs"] if isinstance(row, dict)]
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        # doc_id -> row mapping
        if all(str(key).strip().isdigit() for key in payload.keys()):
            rows: list[dict[str, Any]] = []
            for key, value in payload.items():
                if not isinstance(value, dict):
                    continue
                row = dict(value)
                row.setdefault("doc_id", int(str(key)))
                rows.append(row)
            return rows

        # Raw image-keyed results.json from existing inference pipeline
        if not doc_info_path:
            raise ValueError(
                "Prediction payload appears to be image-keyed results.json. "
                "Pass --doc-info to aggregate page outputs into doc-level predictions."
            )
        doc_info = load_doc_info(doc_info_path)
        basename_index = build_result_basename_index(payload)
        rows: list[dict[str, Any]] = []
        for doc_id in sorted(doc_info):
            page_rows = doc_info[doc_id]
            merged_page_entries: list[dict[str, Any]] = []
            for page_row in page_rows:
                filename = str(page_row.get("filename", "")).strip()
                if not filename:
                    continue
                raw_entry = lookup_result_entry(
                    results=payload,
                    basename_index=basename_index,
                    filename=filename,
                    images_dir=images_dir,
                )
                entry = _normalize_entry(raw_entry)
                if entry:
                    merged_page_entries.append(entry)
            rows.append(_merge_prediction_doc_rows(merged_page_entries, doc_id=doc_id))
        return rows
    raise ValueError("Unsupported predictions payload format.")


def _load_labels(labels_dir: str | Path) -> list[dict[str, Any]]:
    labels: list[dict[str, Any]] = []
    for path in iter_label_files(labels_dir):
        payload = read_json(path)
        if isinstance(payload, dict):
            labels.append(payload)
    return labels


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score TA benchmark predictions.")
    parser.add_argument("--labels-dir", required=True, help="Directory with final label JSON files")
    parser.add_argument("--predictions", required=True, help="Predictions JSON path")
    parser.add_argument("--output", required=True, help="Output scores JSON path")
    parser.add_argument(
        "--doc-info",
        help="Required when --predictions is image-keyed results.json from inference runs.",
    )
    parser.add_argument("--images-dir", help="Optional images root used with --doc-info")
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for region matching")
    parser.add_argument("--model", help="Optional model id/name for report metadata")
    parser.add_argument("--source-run", help="Optional run name/id for report metadata")
    parser.add_argument("--runtime-seconds", type=float, help="Optional runtime seconds metadata")
    parser.add_argument("--cost-per-image", type=float, help="Optional cost-per-image metadata")
    parser.add_argument("--total-cost", type=float, help="Optional total-cost metadata")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    console.print(f"[yellow]{PII_WARNING_BANNER}[/yellow]")

    labels = _load_labels(args.labels_dir)
    if not labels:
        console.print("[red]No labels found.[/red]")
        return 2

    predictions_payload = read_json(args.predictions)
    predictions = _normalize_predictions_payload(
        predictions_payload,
        doc_info_path=args.doc_info,
        images_dir=args.images_dir,
    )

    scores = score_documents(labels, predictions, iou_threshold=float(args.iou_threshold))
    output_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "labels_dir": str(Path(args.labels_dir).expanduser().resolve(strict=False)),
        "predictions": str(Path(args.predictions).expanduser().resolve(strict=False)),
        "doc_info": str(Path(args.doc_info).expanduser().resolve(strict=False)) if args.doc_info else None,
        "iou_threshold": float(args.iou_threshold),
        "run_metadata": {
            "model": str(args.model or "").strip(),
            "source_run": str(args.source_run or "").strip(),
            "runtime_seconds": float(args.runtime_seconds) if args.runtime_seconds is not None else None,
            "cost_per_image": float(args.cost_per_image) if args.cost_per_image is not None else None,
            "total_cost": float(args.total_cost) if args.total_cost is not None else None,
        },
        "scores": scores,
    }
    out_path = write_json(args.output, output_payload)

    summary = Table(title="TA Score Run")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value")
    summary.add_row("Output", str(out_path))
    summary.add_row("Docs scored", str(scores.get("doc_count", 0)))
    region = scores.get("region_detection", {})
    summary.add_row("Region F1", f"{float(region.get('f1', 0.0)):.4f}")
    rubric = scores.get("rubric_scoring", {})
    summary.add_row("Rubric MAE", f"{float(rubric.get('mae', 0.0)):.4f}")
    summary.add_row("Rubric QWK", f"{float(rubric.get('qwk', 0.0)):.4f}")
    console.print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
