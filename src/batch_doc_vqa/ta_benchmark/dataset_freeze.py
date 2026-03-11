#!/usr/bin/env python3
"""Freeze a TA cohort definition and split metadata for instructor pilot datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.table import Table

from .constants import DEFAULT_TA_WORKSPACE_ROOT, PII_WARNING_BANNER
from .io_utils import load_doc_info, load_test_ids, read_json, sha256_file, write_json

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


console = Console()


def _parse_int_csv(raw: Optional[str]) -> set[int]:
    if not raw:
        return set()
    values: set[int] = set()
    for chunk in raw.split(","):
        text = chunk.strip()
        if not text:
            continue
        values.add(int(text))
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze TA benchmark cohort metadata.")
    parser.add_argument("--doc-info", required=True, help="Path to doc_info.csv")
    parser.add_argument("--test-ids", required=True, help="Path to test_ids.csv")
    parser.add_argument(
        "--dataset-manifest",
        help="Optional path to dataset_manifest.json from export flow",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_TA_WORKSPACE_ROOT,
        help=f"Output root for frozen cohort files (default: {DEFAULT_TA_WORKSPACE_ROOT})",
    )
    parser.add_argument(
        "--assignment-ids",
        help="Optional comma-separated assignment IDs to keep.",
    )
    parser.add_argument("--tier2-size", type=int, default=150, help="Tier 2 subset size (default: 150)")
    parser.add_argument("--dev-size", type=int, default=30, help="Calibration/dev subset size (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic splits")
    parser.add_argument(
        "--filter-policy",
        default=(
            "student_id_not_null,required_pages_present,dedupe_latest,"
            "min_docs_per_assignment_10,allow_ambiguous_section_false"
        ),
        help="Opaque filter policy string saved into cohort_definition.json",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    console.print(f"[yellow]{PII_WARNING_BANNER}[/yellow]")

    doc_info_path = Path(args.doc_info).expanduser().resolve(strict=False)
    test_ids_path = Path(args.test_ids).expanduser().resolve(strict=False)
    manifest_path = (
        Path(args.dataset_manifest).expanduser().resolve(strict=False)
        if args.dataset_manifest
        else None
    )
    output_root = Path(args.output_root).expanduser().resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=True)

    doc_info = load_doc_info(doc_info_path)
    gt = load_test_ids(test_ids_path)
    docs = sorted(set(doc_info.keys()) & set(gt.keys()))
    if not docs:
        console.print("[red]No overlapping doc IDs found between doc_info.csv and test_ids.csv[/red]")
        return 2

    assignment_filter = _parse_int_csv(args.assignment_ids)
    if assignment_filter:
        filtered: list[int] = []
        for doc in docs:
            row = gt.get(doc, {})
            try:
                assignment_id = int(str(row.get("assignment_id", "")).strip())
            except ValueError:
                continue
            if assignment_id in assignment_filter:
                filtered.append(doc)
        docs = filtered
        if not docs:
            console.print("[red]All docs were filtered out by --assignment-ids[/red]")
            return 2

    rng = random.Random(int(args.seed))
    docs_for_sampling = list(docs)
    rng.shuffle(docs_for_sampling)

    tier2_size = max(0, int(args.tier2_size))
    dev_size = max(0, int(args.dev_size))

    tier2_docs = sorted(docs_for_sampling[: min(tier2_size, len(docs_for_sampling))])
    dev_docs = sorted(tier2_docs[: min(dev_size, len(tier2_docs))])
    dev_doc_set = set(dev_docs)
    test_docs = sorted(doc for doc in docs if doc not in dev_doc_set)

    hashes = {
        "doc_info_sha256": sha256_file(doc_info_path),
        "test_ids_sha256": sha256_file(test_ids_path),
    }
    manifest_payload = None
    if manifest_path and manifest_path.exists():
        hashes["dataset_manifest_sha256"] = sha256_file(manifest_path)
        manifest_payload = read_json(manifest_path)

    dataset_hash_material = "|".join(
        f"{key}:{value}" for key, value in sorted(hashes.items(), key=lambda item: item[0])
    )
    dataset_hash = hashlib.sha256(dataset_hash_material.encode("utf-8")).hexdigest()

    cohort_definition = {
        "dataset_hash": dataset_hash,
        "doc_info_file": str(doc_info_path),
        "test_ids_file": str(test_ids_path),
        "dataset_manifest_file": str(manifest_path) if manifest_path else None,
        "hashes": hashes,
        "assignment_ids_filter": sorted(assignment_filter),
        "filter_policy": args.filter_policy,
        "seed": int(args.seed),
        "documents_total": len(docs),
        "tier2_size": len(tier2_docs),
        "dev_size": len(dev_docs),
    }
    if manifest_payload is not None:
        cohort_definition["dataset_manifest"] = manifest_payload

    ta_dataset = {
        "name": "instructor_pilot_ta_v1",
        "dataset_hash": dataset_hash,
        "doc_info_file": str(doc_info_path),
        "test_ids_file": str(test_ids_path),
        "dataset_manifest_file": str(manifest_path) if manifest_path else None,
        "splits": {
            "tier1_docs": docs,
            "tier2_docs": tier2_docs,
            "dev_docs": dev_docs,
            "test_docs": test_docs,
        },
    }

    cohort_path = output_root / "cohort_definition.json"
    ta_dataset_path = output_root / "ta_dataset.yaml"
    write_json(cohort_path, cohort_definition)
    with open(ta_dataset_path, "w", encoding="utf-8") as f:
        if yaml is not None:
            yaml.safe_dump(ta_dataset, f, sort_keys=False)
        else:
            # JSON is valid YAML 1.2; this keeps output consumable without PyYAML.
            json.dump(ta_dataset, f, indent=2, ensure_ascii=True)
            f.write("\n")
            console.print(
                "[yellow]PyYAML unavailable; wrote ta_dataset.yaml as JSON-compatible YAML.[/yellow]"
            )

    summary = Table(title="TA Cohort Freeze")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value")
    summary.add_row("Output root", str(output_root))
    summary.add_row("dataset_hash", dataset_hash)
    summary.add_row("tier1_docs", str(len(docs)))
    summary.add_row("tier2_docs", str(len(tier2_docs)))
    summary.add_row("dev_docs", str(len(dev_docs)))
    summary.add_row("test_docs", str(len(test_docs)))
    summary.add_row("cohort_definition.json", str(cohort_path))
    summary.add_row("ta_dataset.yaml", str(ta_dataset_path))
    console.print(summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
