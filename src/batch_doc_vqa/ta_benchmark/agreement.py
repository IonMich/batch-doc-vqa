#!/usr/bin/env python3
"""Compute agreement between two annotation directories."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .io_utils import write_json
from .metrics import score_documents
from .schema import iter_label_files
from .io_utils import read_json


console = Console()


def _load_labels(labels_dir: str | Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    for path in iter_label_files(labels_dir):
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        try:
            doc_id = int(payload.get("doc_id"))
        except (TypeError, ValueError):
            continue
        rows[doc_id] = payload
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check inter-annotator agreement for TA labels.")
    parser.add_argument("--labels-a", required=True, help="Primary label directory")
    parser.add_argument("--labels-b", required=True, help="Secondary label directory")
    parser.add_argument(
        "--threshold-template",
        type=float,
        default=0.85,
        help="Minimum template matching agreement (default: 0.85)",
    )
    parser.add_argument(
        "--threshold-region-f1",
        type=float,
        default=0.80,
        help="Minimum region detection F1 agreement (default: 0.80)",
    )
    parser.add_argument("--iou-threshold", type=float, default=0.5, help="IoU threshold for region agreement")
    parser.add_argument("--output", help="Optional JSON output path")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    labels_a = _load_labels(args.labels_a)
    labels_b = _load_labels(args.labels_b)
    common_doc_ids = sorted(set(labels_a.keys()) & set(labels_b.keys()))
    if not common_doc_ids:
        console.print("[red]No overlapping doc IDs between labels-a and labels-b[/red]")
        return 2

    docs_a = [labels_a[doc_id] for doc_id in common_doc_ids]
    docs_b = [labels_b[doc_id] for doc_id in common_doc_ids]
    scores = score_documents(docs_a, docs_b, iou_threshold=float(args.iou_threshold))

    template_acc = float(scores.get("template_matching", {}).get("top1_accuracy", 0.0))
    region_f1 = float(scores.get("region_detection", {}).get("f1", 0.0))
    passes = (
        template_acc >= float(args.threshold_template)
        and region_f1 >= float(args.threshold_region_f1)
    )

    summary = Table(title="TA Agreement Check")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_column("Threshold", justify="right")
    summary.add_column("Pass", justify="right")
    summary.add_row(
        "Template agreement",
        f"{template_acc:.4f}",
        f"{float(args.threshold_template):.4f}",
        "yes" if template_acc >= float(args.threshold_template) else "no",
    )
    summary.add_row(
        "Region F1 agreement",
        f"{region_f1:.4f}",
        f"{float(args.threshold_region_f1):.4f}",
        "yes" if region_f1 >= float(args.threshold_region_f1) else "no",
    )
    summary.add_row("Common docs", str(len(common_doc_ids)), "-", "-")
    summary.add_row("Overall", "PASS" if passes else "FAIL", "-", "yes" if passes else "no")
    console.print(summary)

    if args.output:
        payload = {
            "labels_a": str(Path(args.labels_a).expanduser().resolve(strict=False)),
            "labels_b": str(Path(args.labels_b).expanduser().resolve(strict=False)),
            "common_docs": common_doc_ids,
            "scores": scores,
            "thresholds": {
                "template": float(args.threshold_template),
                "region_f1": float(args.threshold_region_f1),
            },
            "passes": passes,
        }
        out_path = write_json(args.output, payload)
        console.print(f"[green]Wrote agreement report: {out_path}[/green]")

    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
