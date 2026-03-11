#!/usr/bin/env python3
"""CLI: validate TA benchmark label files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from .constants import DEFAULT_TA_SCHEMA_PATH, DEFAULT_TA_TAXONOMY_PATH, PII_WARNING_BANNER
from .schema import (
    iter_label_files,
    load_error_taxonomy,
    load_ta_schema,
    validate_label_file,
)


console = Console()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate TA benchmark labels.")
    parser.add_argument("--labels-dir", required=True, help="Directory with label JSON files.")
    parser.add_argument(
        "--schema",
        default=DEFAULT_TA_SCHEMA_PATH,
        help=f"Path to TA schema JSON (default: {DEFAULT_TA_SCHEMA_PATH})",
    )
    parser.add_argument(
        "--taxonomy",
        default=DEFAULT_TA_TAXONOMY_PATH,
        help=f"Path to error taxonomy JSON (default: {DEFAULT_TA_TAXONOMY_PATH})",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON output path for full validation report.",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    console.print(f"[yellow]{PII_WARNING_BANNER}[/yellow]")

    labels_dir = Path(args.labels_dir).expanduser().resolve(strict=False)
    schema = load_ta_schema(args.schema)
    known_error_tags = load_error_taxonomy(args.taxonomy)
    files = list(iter_label_files(labels_dir))
    if not files:
        console.print(f"[red]No label JSON files found under {labels_dir}[/red]")
        return 2

    report_rows: list[dict[str, Any]] = []
    valid_count = 0
    issue_count = 0

    for path in files:
        result = validate_label_file(path, schema=schema, known_error_tags=known_error_tags)
        if result.is_valid:
            valid_count += 1
        issue_count += len(result.issues)
        report_rows.append(
            {
                "file": str(path),
                "valid": result.is_valid,
                "issues": [
                    {
                        "level": issue.level,
                        "code": issue.code,
                        "message": issue.message,
                        "path": issue.path,
                    }
                    for issue in result.issues
                ],
            }
        )

    summary = Table(title="TA Label Validation Summary")
    summary.add_column("Metric", style="cyan")
    summary.add_column("Value", justify="right")
    summary.add_row("Label files", str(len(files)))
    summary.add_row("Valid files", str(valid_count))
    summary.add_row("Invalid files", str(len(files) - valid_count))
    summary.add_row("Total issues", str(issue_count))
    console.print(summary)

    invalid_rows = [row for row in report_rows if not row["valid"]]
    if invalid_rows:
        detail = Table(title="Invalid Files (First 20)")
        detail.add_column("File", style="yellow")
        detail.add_column("Issue Count", justify="right")
        detail.add_column("First Issue")
        for row in invalid_rows[:20]:
            issues = row["issues"]
            first_issue = issues[0]["message"] if issues else "Unknown error"
            detail.add_row(row["file"], str(len(issues)), first_issue)
        console.print(detail)

    payload = {
        "labels_dir": str(labels_dir),
        "schema": str(Path(args.schema).expanduser().resolve(strict=False)),
        "taxonomy": str(Path(args.taxonomy).expanduser().resolve(strict=False)),
        "total_files": len(files),
        "valid_files": valid_count,
        "invalid_files": len(files) - valid_count,
        "total_issues": issue_count,
        "rows": report_rows,
    }

    if args.output:
        output_path = Path(args.output).expanduser().resolve(strict=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=True)
        console.print(f"[green]Wrote validation report: {output_path}[/green]")

    return 0 if valid_count == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
