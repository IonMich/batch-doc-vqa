#!/usr/bin/env python3
"""Materialize split-specific doc_info/test_ids files from ta_dataset metadata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from .constants import PII_WARNING_BANNER

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


console = Console()


def _load_ta_dataset(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        if yaml is None:
            raise ValueError(
                "ta_dataset file is YAML but PyYAML is unavailable. Install PyYAML or provide JSON-compatible content."
            )
        payload = yaml.safe_load(text)

    if not isinstance(payload, dict):
        raise ValueError("ta_dataset must be a mapping/object.")
    return payload


def _normalize_doc_ids(raw_docs: Any) -> list[int]:
    if not isinstance(raw_docs, list):
        raise ValueError("split entry must be a list of doc IDs.")

    docs: list[int] = []
    seen: set[int] = set()
    for item in raw_docs:
        try:
            value = int(item)
        except (TypeError, ValueError):
            continue
        if value in seen:
            continue
        seen.add(value)
        docs.append(value)
    return docs


def _filter_csv_by_docs(
    *,
    source_path: Path,
    output_path: Path,
    doc_ids: set[int],
) -> tuple[int, set[int]]:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    found_docs: set[int] = set()
    with open(source_path, "r", encoding="utf-8", newline="") as f_in:
        reader = csv.DictReader(f_in)
        fieldnames = list(reader.fieldnames or [])
        if "doc" not in fieldnames:
            raise ValueError(f"CSV missing required 'doc' column: {source_path}")

        with open(output_path, "w", encoding="utf-8", newline="") as f_out:
            writer = csv.DictWriter(f_out, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                if row is None:
                    continue
                try:
                    doc = int(str(row.get("doc", "")).strip())
                except ValueError:
                    continue
                if doc not in doc_ids:
                    continue
                writer.writerow(row)
                kept += 1
                found_docs.add(doc)

    return kept, found_docs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize split-specific manifest files from ta_dataset.yaml",
    )
    parser.add_argument("--ta-dataset", required=True, help="Path to ta_dataset.yaml/json")
    parser.add_argument(
        "--split",
        default="dev_docs",
        help="Split key in ta_dataset.splits (default: dev_docs)",
    )
    parser.add_argument(
        "--doc-info",
        help="Optional source doc_info.csv override (default: ta_dataset.doc_info_file)",
    )
    parser.add_argument(
        "--test-ids",
        help="Optional source test_ids.csv override (default: ta_dataset.test_ids_file)",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory for subset files (default: <ta_dataset_dir>/subsets)",
    )
    parser.add_argument(
        "--output-doc-info",
        help="Optional output path for subset doc_info.csv",
    )
    parser.add_argument(
        "--output-test-ids",
        help="Optional output path for subset test_ids.csv",
    )
    parser.add_argument(
        "--skip-test-ids",
        action="store_true",
        help="Do not materialize test_ids subset file.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    console.print(f"[yellow]{PII_WARNING_BANNER}[/yellow]")

    ta_dataset_path = Path(args.ta_dataset).expanduser().resolve(strict=False)
    if not ta_dataset_path.exists():
        console.print(f"[red]ta_dataset file does not exist: {ta_dataset_path}[/red]")
        return 2

    payload = _load_ta_dataset(ta_dataset_path)
    splits = payload.get("splits", {})
    if not isinstance(splits, dict):
        console.print("[red]ta_dataset missing 'splits' mapping.[/red]")
        return 2

    if args.split not in splits:
        available = ", ".join(sorted(str(k) for k in splits.keys()))
        console.print(f"[red]Unknown split '{args.split}'. Available: {available}[/red]")
        return 2

    docs = _normalize_doc_ids(splits.get(args.split))
    if not docs:
        console.print(f"[red]Split '{args.split}' has no valid doc IDs.[/red]")
        return 2
    doc_set = set(docs)

    doc_info_src = Path(
        args.doc_info
        or payload.get("doc_info_file", "")
    ).expanduser().resolve(strict=False)
    if not doc_info_src.exists():
        console.print(f"[red]doc_info.csv does not exist: {doc_info_src}[/red]")
        return 2

    test_ids_src: Optional[Path] = None
    if not args.skip_test_ids:
        test_ids_value = args.test_ids or payload.get("test_ids_file", "")
        test_ids_src = Path(test_ids_value).expanduser().resolve(strict=False)
        if not test_ids_src.exists():
            console.print(f"[red]test_ids.csv does not exist: {test_ids_src}[/red]")
            return 2

    output_dir = (
        Path(args.output_dir).expanduser().resolve(strict=False)
        if args.output_dir
        else ta_dataset_path.parent / "subsets"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    output_doc_info = (
        Path(args.output_doc_info).expanduser().resolve(strict=False)
        if args.output_doc_info
        else output_dir / f"doc_info_{args.split}.csv"
    )
    output_test_ids = (
        Path(args.output_test_ids).expanduser().resolve(strict=False)
        if args.output_test_ids
        else output_dir / f"test_ids_{args.split}.csv"
    )

    doc_rows, doc_info_found_docs = _filter_csv_by_docs(
        source_path=doc_info_src,
        output_path=output_doc_info,
        doc_ids=doc_set,
    )
    test_rows = 0
    test_ids_found_docs: set[int] = set()
    if test_ids_src is not None:
        test_rows, test_ids_found_docs = _filter_csv_by_docs(
            source_path=test_ids_src,
            output_path=output_test_ids,
            doc_ids=doc_set,
        )

    missing_from_doc_info = sorted(doc_set - doc_info_found_docs)
    missing_from_test_ids = sorted(doc_set - test_ids_found_docs) if test_ids_src is not None else []

    table = Table(title="TA Split Subset")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Split", str(args.split))
    table.add_row("Requested docs", str(len(doc_set)))
    table.add_row("doc_info source", str(doc_info_src))
    table.add_row("doc_info output", str(output_doc_info))
    table.add_row("doc_info rows kept", str(doc_rows))
    table.add_row("doc_info docs found", str(len(doc_info_found_docs)))
    if test_ids_src is not None:
        table.add_row("test_ids source", str(test_ids_src))
        table.add_row("test_ids output", str(output_test_ids))
        table.add_row("test_ids rows kept", str(test_rows))
        table.add_row("test_ids docs found", str(len(test_ids_found_docs)))
    table.add_row("missing docs in doc_info", str(len(missing_from_doc_info)))
    if test_ids_src is not None:
        table.add_row("missing docs in test_ids", str(len(missing_from_test_ids)))
    console.print(table)

    if missing_from_doc_info:
        console.print(f"[yellow]Missing docs in doc_info: {missing_from_doc_info}[/yellow]")
    if missing_from_test_ids:
        console.print(f"[yellow]Missing docs in test_ids: {missing_from_test_ids}[/yellow]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
