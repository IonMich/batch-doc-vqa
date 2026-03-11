#!/usr/bin/env python3
"""Generate markdown and JSON reports from TA scoring output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .io_utils import read_json, write_json


console = Console()


def _metric_row(name: str, value: Any) -> tuple[str, str]:
    if isinstance(value, float):
        return name, f"{value:.4f}"
    return name, str(value)


def build_markdown_report(payload: dict[str, Any]) -> str:
    scores = payload.get("scores", {}) if isinstance(payload, dict) else {}
    run_meta = payload.get("run_metadata", {}) if isinstance(payload, dict) else {}
    region = scores.get("region_detection", {})
    transcription = scores.get("description_transcription", {})
    figures = scores.get("figure_association", {})
    template = scores.get("template_matching", {})
    tags = scores.get("error_tagging", {})
    rubric = scores.get("rubric_scoring", {})
    feedback = scores.get("feedback_quality", {})

    lines: list[str] = []
    lines.append("# TA Benchmark Report")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- **Generated at (UTC)**: `{payload.get('generated_at_utc', '')}`")
    lines.append(f"- **Labels dir**: `{payload.get('labels_dir', '')}`")
    lines.append(f"- **Predictions file**: `{payload.get('predictions', '')}`")
    if payload.get("doc_info"):
        lines.append(f"- **Doc manifest**: `{payload.get('doc_info')}`")
    model = str(run_meta.get("model", "") or "").strip()
    source_run = str(run_meta.get("source_run", "") or "").strip()
    if model:
        lines.append(f"- **Model**: `{model}`")
    if source_run:
        lines.append(f"- **Source run**: `{source_run}`")
    if run_meta.get("runtime_seconds") is not None:
        lines.append(f"- **Runtime (seconds)**: `{float(run_meta.get('runtime_seconds')):.3f}`")
    if run_meta.get("cost_per_image") is not None:
        lines.append(f"- **Cost per image**: `${float(run_meta.get('cost_per_image')):.8f}`")
    if run_meta.get("total_cost") is not None:
        lines.append(f"- **Total cost**: `${float(run_meta.get('total_cost')):.6f}`")
    lines.append(f"- **Documents scored**: **{scores.get('doc_count', 0)}**")
    lines.append("")
    lines.append("## Metric Summary")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | --- |")

    metric_rows = [
        _metric_row("Region Precision", region.get("precision", 0.0)),
        _metric_row("Region Recall", region.get("recall", 0.0)),
        _metric_row("Region F1", region.get("f1", 0.0)),
        _metric_row("Description CER", transcription.get("cer", 0.0)),
        _metric_row("Description Norm Edit", transcription.get("normalized_edit_distance", 0.0)),
        _metric_row("Figure Link F1", figures.get("f1", 0.0)),
        _metric_row("Template Top-1 Accuracy", template.get("top1_accuracy", 0.0)),
        _metric_row("Error Tag Micro F1", tags.get("micro_f1", 0.0)),
        _metric_row("Error Tag Macro F1", tags.get("macro_f1", 0.0)),
        _metric_row("Rubric MAE", rubric.get("mae", 0.0)),
        _metric_row("Rubric Exact Match Rate", rubric.get("exact_match_rate", 0.0)),
        _metric_row("Rubric QWK", rubric.get("qwk", 0.0)),
        _metric_row("Feedback Correctness Agreement", feedback.get("correctness_agreement", 0.0)),
        _metric_row("Feedback Specificity Agreement", feedback.get("specificity_agreement", 0.0)),
        _metric_row("Feedback Actionability Agreement", feedback.get("actionability_agreement", 0.0)),
        _metric_row("Feedback Overall Agreement", feedback.get("overall_agreement", 0.0)),
    ]
    for key, value in metric_rows:
        lines.append(f"| {key} | {value} |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- This report is dataset-scoped and intended for TA benchmark comparison runs.")
    lines.append("- Keep raw instructor-pilot data local and out of version control.")
    lines.append("")

    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate TA benchmark report from scores JSON.")
    parser.add_argument("--scores-json", required=True, help="Input scores JSON (from ta-score-runs)")
    parser.add_argument("--output-md", required=True, help="Output markdown report path")
    parser.add_argument("--output-json", help="Optional output JSON path (normalized report payload)")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    payload = read_json(args.scores_json)
    if not isinstance(payload, dict):
        console.print("[red]scores-json must be an object[/red]")
        return 2

    markdown = build_markdown_report(payload)
    output_md = Path(args.output_md).expanduser().resolve(strict=False)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(markdown, encoding="utf-8")

    if args.output_json:
        normalized_payload = {
            "report_version": "ta_report_v1",
            "source_scores": str(Path(args.scores_json).expanduser().resolve(strict=False)),
            "metrics": payload.get("scores", {}),
        }
        write_json(args.output_json, normalized_payload)

    table = Table(title="TA Report Generation")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("scores-json", str(Path(args.scores_json).expanduser().resolve(strict=False)))
    table.add_row("output-md", str(output_md))
    table.add_row("output-json", str(Path(args.output_json).expanduser().resolve(strict=False)) if args.output_json else "N/A")
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
