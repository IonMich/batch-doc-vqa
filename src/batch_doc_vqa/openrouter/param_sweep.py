#!/usr/bin/env python3
"""
Parameter sweep CLI for OpenRouter inference.

Runs one or more generation-parameter conditions on the same dataset and reports
how quality, token usage, costs, and repetition behavior shift across conditions.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from rich.console import Console
from rich.table import Table

from ..core.run_manager import RunManager
from .cli import parse_pages, parse_provider_order
from .defaults import (
    DEFAULT_DATASET_MANIFEST_FILE,
    DEFAULT_EXTRACTION_PRESET_ID,
    DEFAULT_IMAGES_DIR,
)
from .presets import available_preset_ids, resolve_preset_definition


console = Console()

SWEEP_PARAM_PARSERS = {
    "temperature": float,
    "max_tokens": int,
    "top_p": float,
    "top_k": int,
    "min_p": float,
    "presence_penalty": float,
    "repetition_penalty": float,
}

SWEEPABLE_PARAM_NAMES = tuple(SWEEP_PARAM_PARSERS.keys())

FAILURE_MARKERS = {
    "_schema_failed",
    "_parse_failed",
    "_no_response",
    "_empty_response",
    "_server_error",
    "_api_error",
    "_exception",
    "_retry_failed",
}

QUALITY_METRIC_KEYS = (
    "digit_top1",
    "id_top1",
    "lastname_top1",
    "id_avg_lev",
    "lastname_avg_lev",
    "docs_detected",
    "docs_detected_count",
)

USAGE_METRIC_KEYS = (
    "images_total",
    "images_success",
    "images_failed",
    "failure_rate",
    "tokens_prompt_total",
    "tokens_completion_total",
    "tokens_total",
    "tokens_prompt_avg",
    "tokens_completion_avg",
    "tokens_total_avg",
    "actual_cost_total",
    "actual_cost_avg_per_image",
    "actual_cost_avg_scored",
    "repetition_count",
    "repetition_rate",
    "repetition_score_avg",
    "repetition_score_max",
)

AGGREGATABLE_METRIC_KEYS = QUALITY_METRIC_KEYS + USAGE_METRIC_KEYS + (
    "runtime_seconds",
)


@dataclass(frozen=True)
class SweepAxis:
    """One sweep axis, e.g. temperature=[0.0, 1.0]."""

    name: str
    values: tuple[Any, ...]


def _coerce_numeric(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _is_failure_entry(entry: Dict[str, Any]) -> bool:
    for marker in FAILURE_MARKERS:
        marker_value = entry.get(marker)
        if marker_value not in (None, False, 0, ""):
            return True
    return False


def parse_sweep_axis(raw_spec: str) -> SweepAxis:
    """
    Parse one axis spec.

    Example:
      temperature=0,1
      top_k=10,20,40
    """
    if "=" not in raw_spec:
        raise ValueError(
            f"Invalid sweep spec {raw_spec!r}. Expected format: <param>=v1,v2,..."
        )

    key_raw, values_raw = raw_spec.split("=", 1)
    key = key_raw.strip().lower().replace("-", "_")
    parser = SWEEP_PARAM_PARSERS.get(key)
    if parser is None:
        supported = ", ".join(SWEEPABLE_PARAM_NAMES)
        raise ValueError(
            f"Unsupported sweep param {key!r}. Supported: {supported}"
        )

    values: list[Any] = []
    seen: set[Any] = set()
    for chunk in values_raw.split(","):
        token = chunk.strip()
        if not token:
            continue
        try:
            parsed_value = parser(token)
        except ValueError as exc:
            raise ValueError(
                f"Invalid value {token!r} for {key}. "
                f"Expected {parser.__name__}."
            ) from exc
        if parsed_value in seen:
            continue
        seen.add(parsed_value)
        values.append(parsed_value)

    if not values:
        raise ValueError(
            f"Sweep param {key!r} has no valid values. "
            "Use at least one comma-separated value."
        )

    return SweepAxis(name=key, values=tuple(values))


def build_sweep_axes(
    *,
    set_specs: Sequence[str],
    temperature_values: Optional[str],
) -> list[SweepAxis]:
    """Build sweep axes from generic specs + temperature convenience shortcut."""
    axes: list[SweepAxis] = []
    if temperature_values is not None:
        axes.append(parse_sweep_axis(f"temperature={temperature_values}"))
    for spec in set_specs:
        axes.append(parse_sweep_axis(spec))

    if not axes:
        raise ValueError(
            "No sweep parameters were provided. "
            "Use --set <param>=... or --temperature-values ... ."
        )

    seen_names: set[str] = set()
    deduped_axes: list[SweepAxis] = []
    for axis in axes:
        if axis.name in seen_names:
            raise ValueError(
                f"Duplicate sweep axis for {axis.name!r}. "
                "Each parameter can only be specified once."
            )
        seen_names.add(axis.name)
        deduped_axes.append(axis)

    return deduped_axes


def build_condition_grid(axes: Sequence[SweepAxis]) -> list[dict[str, Any]]:
    """Build Cartesian product across sweep axes."""
    if not axes:
        return [{}]

    names = [axis.name for axis in axes]
    value_sets = [axis.values for axis in axes]
    grid: list[dict[str, Any]] = []
    for combo in itertools.product(*value_sets):
        grid.append({name: value for name, value in zip(names, combo)})
    return grid


def format_condition_label(
    condition: Dict[str, Any],
    *,
    axis_order: Iterable[str],
) -> str:
    if not condition:
        return "default"
    parts: list[str] = []
    for key in axis_order:
        if key not in condition:
            continue
        parts.append(f"{key}={condition[key]}")
    return ", ".join(parts) if parts else "default"


def summarize_result_payload(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Summarize token/repetition/failure metrics from raw run results.json payload.
    """
    images_total = float(len(results))
    if images_total <= 0:
        return {
            "images_total": 0.0,
            "images_success": 0.0,
            "images_failed": 0.0,
            "failure_rate": 0.0,
            "tokens_prompt_total": 0.0,
            "tokens_completion_total": 0.0,
            "tokens_total": 0.0,
            "tokens_prompt_avg": 0.0,
            "tokens_completion_avg": 0.0,
            "tokens_total_avg": 0.0,
            "actual_cost_total": 0.0,
            "actual_cost_avg_per_image": 0.0,
            "actual_cost_avg_scored": 0.0,
            "repetition_count": 0.0,
            "repetition_rate": 0.0,
            "repetition_score_avg": 0.0,
            "repetition_score_max": 0.0,
        }

    prompt_total = 0.0
    completion_total = 0.0
    tokens_total = 0.0
    cost_total = 0.0
    cost_count = 0.0
    repetition_count = 0.0
    repetition_scores: list[float] = []
    failed = 0.0

    for result_entries in results.values():
        entry: Dict[str, Any] = {}
        if isinstance(result_entries, list) and result_entries and isinstance(result_entries[0], dict):
            entry = result_entries[0]
        else:
            failed += 1.0
            continue

        if _is_failure_entry(entry):
            failed += 1.0

        token_usage = entry.get("_token_usage")
        if isinstance(token_usage, dict):
            prompt_total += _coerce_numeric(token_usage.get("prompt_tokens")) or 0.0
            completion_total += _coerce_numeric(token_usage.get("completion_tokens")) or 0.0
            tokens_total += _coerce_numeric(token_usage.get("total_tokens")) or 0.0
            maybe_cost = _coerce_numeric(token_usage.get("actual_cost"))
            if maybe_cost is not None:
                cost_total += maybe_cost
                cost_count += 1.0

        if entry.get("_repetition_detected"):
            repetition_count += 1.0
            score = _coerce_numeric(entry.get("_repetition_score")) or 0.0
            repetition_scores.append(score)

    success = images_total - failed
    repetition_rate = 100.0 * repetition_count / max(1.0, images_total)
    failure_rate = 100.0 * failed / max(1.0, images_total)

    return {
        "images_total": images_total,
        "images_success": success,
        "images_failed": failed,
        "failure_rate": failure_rate,
        "tokens_prompt_total": prompt_total,
        "tokens_completion_total": completion_total,
        "tokens_total": tokens_total,
        "tokens_prompt_avg": prompt_total / max(1.0, images_total),
        "tokens_completion_avg": completion_total / max(1.0, images_total),
        "tokens_total_avg": tokens_total / max(1.0, images_total),
        "actual_cost_total": cost_total,
        "actual_cost_avg_per_image": cost_total / max(1.0, images_total),
        "actual_cost_avg_scored": cost_total / max(1.0, cost_count),
        "repetition_count": repetition_count,
        "repetition_rate": repetition_rate,
        "repetition_score_avg": (
            statistics.mean(repetition_scores) if repetition_scores else 0.0
        ),
        "repetition_score_max": max(repetition_scores) if repetition_scores else 0.0,
    }


def _find_run_info(run_manager: RunManager, run_name: str) -> Dict[str, Any]:
    escaped = re.escape(run_name)
    matches = run_manager.list_runs(pattern=f"^{escaped}$")
    for run in matches:
        if run.get("run_name") == run_name:
            return run
    raise FileNotFoundError(f"Run metadata not found for {run_name}")


def summarize_run(
    *,
    run_manager: RunManager,
    benchmark_generator: Any,
    run_name: str,
    doc_info_file: str,
    test_ids_file: str,
) -> Dict[str, Any]:
    run_info = _find_run_info(run_manager, run_name)
    results = run_manager.load_results(run_name)
    usage_stats = summarize_result_payload(results)
    quality_stats = benchmark_generator.compute_run_stats(
        run_info,
        doc_info_file=doc_info_file,
        test_ids_file=test_ids_file,
    ) or {}

    config = run_info.get("config", {})
    additional = config.get("additional", {}) if isinstance(config, dict) else {}
    environment = config.get("environment", {}) if isinstance(config, dict) else {}
    api = config.get("api", {}) if isinstance(config, dict) else {}

    provider_list = []
    if isinstance(additional, dict):
        providers_raw = additional.get("actual_model_providers")
        if isinstance(providers_raw, list):
            provider_list = [str(v) for v in providers_raw]

    runtime_seconds = None
    if isinstance(additional, dict):
        runtime_seconds = _coerce_numeric(additional.get("actual_runtime_seconds"))

    summary: Dict[str, Any] = {
        "run_name": run_name,
        "providers": provider_list,
        "runtime": environment.get("runtime"),
        "runtime_seconds": runtime_seconds,
        "effective_api_params": api if isinstance(api, dict) else {},
    }

    for key in QUALITY_METRIC_KEYS:
        summary[key] = _coerce_numeric(quality_stats.get(key))
    summary.update(usage_stats)
    return summary


def aggregate_condition_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Aggregate trial rows by condition label.
    Uses arithmetic mean per numeric metric.
    """
    grouped: Dict[str, list[Dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        label = str(row["condition_label"])
        if label not in grouped:
            grouped[label] = []
            order.append(label)
        grouped[label].append(row)

    aggregated: list[Dict[str, Any]] = []
    for label in order:
        bucket = grouped[label]
        condition = bucket[0].get("condition", {})
        successes = [row for row in bucket if row.get("summary") is not None]
        failures = len(bucket) - len(successes)
        agg: Dict[str, Any] = {
            "condition_label": label,
            "condition": condition,
            "trials_total": len(bucket),
            "trials_succeeded": len(successes),
            "trials_failed": failures,
            "run_names": [row.get("run_name") for row in successes if row.get("run_name")],
        }
        for key in AGGREGATABLE_METRIC_KEYS:
            values: list[float] = []
            for row in successes:
                summary = row.get("summary")
                if not isinstance(summary, dict):
                    continue
                maybe_num = _coerce_numeric(summary.get(key))
                if maybe_num is not None:
                    values.append(maybe_num)
            agg[key] = statistics.mean(values) if values else None
        aggregated.append(agg)
    return aggregated


def _fmt_pct(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.2f}%"


def _fmt_num(value: Optional[float], decimals: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{decimals}f}"


def _fmt_currency(value: Optional[float], decimals: int = 6) -> str:
    return "N/A" if value is None else f"${value:.{decimals}f}"


def _fmt_delta(value: Optional[float], baseline: Optional[float], decimals: int = 2, suffix: str = "") -> str:
    if value is None or baseline is None:
        return "N/A"
    delta = value - baseline
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.{decimals}f}{suffix}"


def print_aggregate_table(rows: List[Dict[str, Any]]) -> None:
    table = Table(title="Parameter Sweep Summary")
    table.add_column("Condition", style="cyan")
    table.add_column("Trials", justify="right")
    table.add_column("ID Top-1", justify="right")
    table.add_column("Digit Top-1", justify="right")
    table.add_column("Docs detected", justify="right")
    table.add_column("Avg tokens/img", justify="right")
    table.add_column("Repetition", justify="right")
    table.add_column("Failure", justify="right")
    table.add_column("Cost/img", justify="right")
    table.add_column("Runtime (s)", justify="right")

    for row in rows:
        table.add_row(
            row["condition_label"],
            f"{int(row['trials_succeeded'])}/{int(row['trials_total'])}",
            _fmt_pct(_coerce_numeric(row.get("id_top1"))),
            _fmt_pct(_coerce_numeric(row.get("digit_top1"))),
            _fmt_pct(_coerce_numeric(row.get("docs_detected"))),
            _fmt_num(_coerce_numeric(row.get("tokens_total_avg")), decimals=1),
            _fmt_pct(_coerce_numeric(row.get("repetition_rate"))),
            _fmt_pct(_coerce_numeric(row.get("failure_rate"))),
            _fmt_currency(_coerce_numeric(row.get("actual_cost_avg_per_image")), decimals=6),
            _fmt_num(_coerce_numeric(row.get("runtime_seconds")), decimals=1),
        )
    console.print(table)


def print_delta_table(rows: List[Dict[str, Any]], *, baseline_index: int) -> None:
    if len(rows) <= 1:
        return
    if baseline_index < 1 or baseline_index > len(rows):
        raise ValueError(
            f"--baseline-condition-index must be between 1 and {len(rows)}"
        )

    baseline = rows[baseline_index - 1]
    table = Table(
        title=(
            "Delta vs Baseline "
            f"(#{baseline_index}: {baseline['condition_label']})"
        )
    )
    table.add_column("Condition", style="cyan")
    table.add_column("d ID Top-1", justify="right")
    table.add_column("d Docs detected", justify="right")
    table.add_column("d Avg tokens/img", justify="right")
    table.add_column("d Repetition", justify="right")
    table.add_column("d Failure", justify="right")
    table.add_column("d Cost/img", justify="right")

    for row in rows:
        table.add_row(
            row["condition_label"],
            _fmt_delta(
                _coerce_numeric(row.get("id_top1")),
                _coerce_numeric(baseline.get("id_top1")),
                suffix="%",
            ),
            _fmt_delta(
                _coerce_numeric(row.get("docs_detected")),
                _coerce_numeric(baseline.get("docs_detected")),
                suffix="%",
            ),
            _fmt_delta(
                _coerce_numeric(row.get("tokens_total_avg")),
                _coerce_numeric(baseline.get("tokens_total_avg")),
                decimals=1,
            ),
            _fmt_delta(
                _coerce_numeric(row.get("repetition_rate")),
                _coerce_numeric(baseline.get("repetition_rate")),
                suffix="%",
            ),
            _fmt_delta(
                _coerce_numeric(row.get("failure_rate")),
                _coerce_numeric(baseline.get("failure_rate")),
                suffix="%",
            ),
            _fmt_delta(
                _coerce_numeric(row.get("actual_cost_avg_per_image")),
                _coerce_numeric(baseline.get("actual_cost_avg_per_image")),
                decimals=6,
            ),
        )
    console.print(table)


def _resolve_strict_schema_flag(args: argparse.Namespace) -> Optional[bool]:
    if args.strict_schema:
        return True
    if args.no_strict_schema:
        return False
    return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run parameter sweeps for OpenRouter extraction and compare quality, "
            "token usage, cost, and repetition metrics."
        )
    )
    parser.add_argument("--model", required=True, help="OpenRouter model id (org/model)")
    parser.add_argument(
        "--set",
        dest="set_specs",
        action="append",
        default=[],
        help=(
            "Sweep spec in the form <param>=v1,v2,... . "
            f"Supported params: {', '.join(SWEEPABLE_PARAM_NAMES)}. "
            "Repeat for Cartesian sweeps."
        ),
    )
    parser.add_argument(
        "--temperature-values",
        type=str,
        help="Shortcut for --set temperature=v1,v2,...",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of repeated runs per condition (default: 1).",
    )
    parser.add_argument(
        "--max-conditions",
        type=int,
        default=64,
        help="Hard limit on generated condition count (default: 64).",
    )
    parser.add_argument(
        "--baseline-condition-index",
        type=int,
        default=1,
        help="1-based condition index used as delta baseline (default: 1).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue remaining conditions when one run fails.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conditions and exit without running inference.",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        help="Optional output path for a full sweep report JSON.",
    )

    # Base generation settings (can be fixed while sweeping others)
    parser.add_argument("--temperature", type=float, default=None, help="Fixed temperature (if not swept).")
    parser.add_argument("--max-tokens", type=int, default=None, help="Fixed max_tokens (if not swept).")
    parser.add_argument("--top-p", type=float, default=None, help="Fixed top_p (if not swept).")
    parser.add_argument("--top-k", type=int, default=None, help="Fixed top_k (if not swept).")
    parser.add_argument("--min-p", type=float, default=None, help="Fixed min_p (if not swept).")
    parser.add_argument("--presence-penalty", type=float, default=None, help="Fixed presence_penalty (if not swept).")
    parser.add_argument("--repetition-penalty", type=float, default=None, help="Fixed repetition_penalty (if not swept).")

    parser.add_argument(
        "--preset",
        type=str,
        default=DEFAULT_EXTRACTION_PRESET_ID,
        help=(
            "Extraction preset id "
            f"(default: {DEFAULT_EXTRACTION_PRESET_ID}; "
            f"available: {', '.join(available_preset_ids())})"
        ),
    )
    parser.add_argument(
        "--images-dir",
        type=str,
        default=DEFAULT_IMAGES_DIR,
        help=f"Images directory (default: {DEFAULT_IMAGES_DIR})",
    )
    parser.add_argument(
        "--dataset-manifest",
        "--doc-info",
        dest="dataset_manifest",
        type=str,
        default=DEFAULT_DATASET_MANIFEST_FILE,
        help="Optional dataset manifest CSV (doc,page,filename).",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default=None,
        help="Comma-separated page numbers to include.",
    )
    parser.add_argument("--prompt-file", type=str, help="Optional prompt text file.")
    parser.add_argument("--schema-file", type=str, help="Optional schema JSON file.")
    parser.add_argument("--strict-schema", action="store_true", help="Fail images that violate schema.")
    parser.add_argument("--no-strict-schema", action="store_true", help="Allow schema-violating outputs.")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel request count (default: 1).")
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=None,
        help="Max requests/sec across threads.",
    )
    parser.add_argument("--retry-max", type=int, default=3, help="Max retries for transient errors.")
    parser.add_argument(
        "--retry-base-delay",
        type=float,
        default=2.0,
        help="Retry base delay in seconds.",
    )
    parser.add_argument(
        "--skip-reproducibility-checks",
        action="store_true",
        help="Bypass dirty-tree reproducibility guard for sweep automation.",
    )

    parser.add_argument(
        "--provider-order",
        type=str,
        help="Comma-separated provider slugs in preferred order.",
    )
    parser.add_argument(
        "--no-fallbacks",
        action="store_true",
        help="Disable provider fallbacks.",
    )
    parser.add_argument(
        "--provider-sort",
        choices=["price", "throughput", "latency"],
        help="Provider sorting preference.",
    )
    parser.add_argument(
        "--provider-data-collection",
        choices=["allow", "deny"],
        help="Set provider.data_collection routing constraint.",
    )
    parser.add_argument(
        "--provider-zdr",
        dest="provider_zdr",
        action="store_true",
        help="Restrict routing to ZDR endpoints.",
    )
    parser.add_argument(
        "--no-provider-zdr",
        dest="provider_zdr",
        action="store_false",
        help="Explicitly disable ZDR restriction.",
    )
    parser.set_defaults(provider_zdr=None)

    # Benchmark scoring inputs
    parser.add_argument(
        "--runs-dir",
        default="tests/output/param_sweeps",
        help="Run root directory for storing sweep runs and collecting metrics.",
    )
    parser.add_argument(
        "--doc-info-file",
        default="imgs/q11/doc_info.csv",
        help="Doc metadata CSV used for benchmark quality scoring.",
    )
    parser.add_argument(
        "--test-ids-file",
        default="tests/data/test_ids.csv",
        help="Ground-truth IDs CSV used for benchmark quality scoring.",
    )

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.trials <= 0:
        parser.error("--trials must be > 0")
    if args.max_conditions <= 0:
        parser.error("--max-conditions must be > 0")
    if args.strict_schema and args.no_strict_schema:
        parser.error("--strict-schema and --no-strict-schema are mutually exclusive")

    try:
        preset_definition = resolve_preset_definition(args.preset)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    try:
        pages = parse_pages(args.pages, default_selection=list(preset_definition.default_pages))
        sweep_axes = build_sweep_axes(
            set_specs=args.set_specs,
            temperature_values=args.temperature_values,
        )
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    conditions = build_condition_grid(sweep_axes)
    if len(conditions) > args.max_conditions:
        parser.error(
            f"Generated {len(conditions)} conditions, exceeding --max-conditions={args.max_conditions}."
        )
        return 2
    if args.baseline_condition_index < 1 or args.baseline_condition_index > len(conditions):
        parser.error(
            f"--baseline-condition-index must be between 1 and {len(conditions)}"
        )
        return 2

    axis_order = [axis.name for axis in sweep_axes]
    provider_order = parse_provider_order(args.provider_order)
    allow_fallbacks = False if args.no_fallbacks else None
    strict_schema = _resolve_strict_schema_flag(args)

    base_generation_params = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "min_p": args.min_p,
        "presence_penalty": args.presence_penalty,
        "repetition_penalty": args.repetition_penalty,
    }

    preview_table = Table(title="Planned Sweep Conditions")
    preview_table.add_column("#", justify="right")
    preview_table.add_column("Condition", style="cyan")
    for idx, condition in enumerate(conditions, start=1):
        preview_table.add_row(
            str(idx),
            format_condition_label(condition, axis_order=axis_order),
        )
    console.print(preview_table)

    if args.dry_run:
        return 0

    try:
        from ..benchmarks.table_generator import BenchmarkTableGenerator
    except Exception as exc:
        console.print(
            "[red]❌ Benchmark scorer could not be loaded. "
            "Ensure optional scoring dependencies are installed.[/red]"
        )
        console.print(f"[red]{type(exc).__name__}: {exc}[/red]")
        return 2

    from .inference import run_openrouter_inference

    run_manager = RunManager(args.runs_dir)
    benchmark_generator = BenchmarkTableGenerator(
        runs_base_dir=args.runs_dir,
        interactive=False,
    )

    trial_rows: list[Dict[str, Any]] = []

    for cond_idx, condition in enumerate(conditions, start=1):
        condition_label = format_condition_label(condition, axis_order=axis_order)
        for trial_idx in range(1, args.trials + 1):
            generation_params = dict(base_generation_params)
            generation_params.update(condition)
            console.print(
                f"\n[bold cyan]Condition {cond_idx}/{len(conditions)} "
                f"(trial {trial_idx}/{args.trials})[/bold cyan]: {condition_label}"
            )
            console.print(f"[dim]Generation params override: {generation_params}[/dim]")

            run_name = run_openrouter_inference(
                model_name=args.model,
                preset_id=preset_definition.preset_id,
                temperature=generation_params["temperature"],
                max_tokens=generation_params["max_tokens"],
                top_p=generation_params["top_p"],
                top_k=generation_params["top_k"],
                min_p=generation_params["min_p"],
                presence_penalty=generation_params["presence_penalty"],
                repetition_penalty=generation_params["repetition_penalty"],
                provider_order=provider_order if provider_order else None,
                provider_allow_fallbacks=allow_fallbacks,
                provider_sort=args.provider_sort,
                provider_data_collection=args.provider_data_collection,
                provider_zdr=args.provider_zdr,
                interactive=False,
                confirm_reproducibility_warnings=False,
                skip_reproducibility_checks=args.skip_reproducibility_checks,
                concurrency=args.concurrency,
                rate_limit=args.rate_limit,
                retry_max=args.retry_max,
                retry_base_delay=args.retry_base_delay,
                images_dir=args.images_dir,
                dataset_manifest_file=args.dataset_manifest,
                pages=pages,
                prompt_file=args.prompt_file,
                schema_file=args.schema_file,
                strict_schema=strict_schema,
                output_json=None,
                runs_base_dir=args.runs_dir,
            )

            if not run_name:
                message = f"Inference failed for condition '{condition_label}', trial {trial_idx}."
                console.print(f"[red]❌ {message}[/red]")
                trial_rows.append(
                    {
                        "condition_index": cond_idx,
                        "trial_index": trial_idx,
                        "condition_label": condition_label,
                        "condition": dict(condition),
                        "run_name": None,
                        "summary": None,
                        "error": message,
                    }
                )
                if not args.continue_on_error:
                    console.print("[yellow]Stopping early. Re-run with --continue-on-error to keep going.[/yellow]")
                    aggregated_rows = aggregate_condition_rows(trial_rows)
                    if aggregated_rows:
                        print_aggregate_table(aggregated_rows)
                    return 2
                continue

            try:
                summary = summarize_run(
                    run_manager=run_manager,
                    benchmark_generator=benchmark_generator,
                    run_name=run_name,
                    doc_info_file=args.doc_info_file,
                    test_ids_file=args.test_ids_file,
                )
            except Exception as exc:
                message = f"Failed to summarize run {run_name}: {type(exc).__name__}: {exc}"
                console.print(f"[red]❌ {message}[/red]")
                trial_rows.append(
                    {
                        "condition_index": cond_idx,
                        "trial_index": trial_idx,
                        "condition_label": condition_label,
                        "condition": dict(condition),
                        "run_name": run_name,
                        "summary": None,
                        "error": message,
                    }
                )
                if not args.continue_on_error:
                    console.print("[yellow]Stopping early. Re-run with --continue-on-error to keep going.[/yellow]")
                    aggregated_rows = aggregate_condition_rows(trial_rows)
                    if aggregated_rows:
                        print_aggregate_table(aggregated_rows)
                    return 2
                continue

            trial_rows.append(
                {
                    "condition_index": cond_idx,
                    "trial_index": trial_idx,
                    "condition_label": condition_label,
                    "condition": dict(condition),
                    "run_name": run_name,
                    "summary": summary,
                    "error": None,
                }
            )

    aggregated_rows = aggregate_condition_rows(trial_rows)
    if not aggregated_rows:
        console.print("[red]❌ Sweep finished with no summarized runs.[/red]")
        return 2

    print_aggregate_table(aggregated_rows)
    print_delta_table(
        aggregated_rows,
        baseline_index=args.baseline_condition_index,
    )

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve(strict=False)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_payload = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "preset_id": preset_definition.preset_id,
            "sweep_axes": [
                {"name": axis.name, "values": list(axis.values)}
                for axis in sweep_axes
            ],
            "conditions": conditions,
            "trials": int(args.trials),
            "base_generation_params": base_generation_params,
            "dataset": {
                "images_dir": args.images_dir,
                "dataset_manifest": args.dataset_manifest,
                "pages": pages,
                "doc_info_file": args.doc_info_file,
                "test_ids_file": args.test_ids_file,
            },
            "trial_rows": trial_rows,
            "aggregated_rows": aggregated_rows,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_payload, f, indent=2, ensure_ascii=False)
        console.print(f"[green]✅ Wrote sweep report: {output_path}[/green]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
