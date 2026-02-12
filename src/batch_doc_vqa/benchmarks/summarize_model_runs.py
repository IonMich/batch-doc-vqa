#!/usr/bin/env python3
"""Summarize recent runs for one model within a time window."""

from __future__ import annotations

import argparse
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from rich.console import Console
from rich.table import Table

from ..core.run_manager import RunManager
from .cohorts import get_run_timestamp
from .table_generator import BenchmarkTableGenerator


@dataclass
class WindowSelection:
    anchor_run: Dict[str, Any]
    anchor_timestamp: datetime
    runs_in_window: List[Dict[str, Any]]
    runs_missing_timestamp: List[Dict[str, Any]]


def resolve_window_hours(window_hours: float, window_minutes: Optional[float]) -> float:
    """Resolve final window duration in hours."""
    if window_minutes is not None:
        if window_minutes <= 0:
            raise ValueError("--window-minutes must be > 0")
        return float(window_minutes) / 60.0
    if window_hours <= 0:
        raise ValueError("--window-hours must be > 0")
    return float(window_hours)


def model_key_from_config(config: Dict[str, Any]) -> str:
    """Build canonical model key from run config."""
    model = config.get("model", {})
    org = model.get("org", "unknown")
    model_name = model.get("model", "unknown")
    variant = model.get("variant")
    key = f"{org}/{model_name}"
    if isinstance(variant, str) and variant.strip():
        key += f"-{variant}"
    return key


def filter_model_runs(all_runs: List[Dict[str, Any]], model_key: str) -> List[Dict[str, Any]]:
    """Filter runs by canonical model key (case-insensitive exact match)."""
    needle = model_key.strip().lower()
    return [
        run
        for run in all_runs
        if model_key_from_config(run.get("config", {})).lower() == needle
    ]


def select_window_runs(model_runs: List[Dict[str, Any]], window_hours: float) -> WindowSelection:
    """Pick anchor run and include runs within the lookback window."""
    if not model_runs:
        raise ValueError("No runs provided")

    runs_with_ts: List[Tuple[Dict[str, Any], datetime]] = []
    runs_missing_ts: List[Dict[str, Any]] = []
    for run in model_runs:
        ts = get_run_timestamp(run)
        if ts is None:
            runs_missing_ts.append(run)
            continue
        runs_with_ts.append((run, ts))

    if not runs_with_ts:
        raise ValueError("No timestamped runs found for model")

    runs_with_ts.sort(key=lambda item: item[1], reverse=True)
    anchor_run, anchor_ts = runs_with_ts[0]
    max_age_seconds = window_hours * 3600.0

    in_window: List[Dict[str, Any]] = []
    for run, ts in runs_with_ts:
        age_seconds = (anchor_ts - ts).total_seconds()
        if age_seconds < 0:
            continue
        if age_seconds <= max_age_seconds:
            in_window.append(run)

    return WindowSelection(
        anchor_run=anchor_run,
        anchor_timestamp=anchor_ts,
        runs_in_window=in_window,
        runs_missing_timestamp=runs_missing_ts,
    )


def summarize_values(values: List[float]) -> Dict[str, float]:
    """Compute compact summary stats for a numeric metric."""
    if not values:
        return {}
    result: Dict[str, float] = {
        "n": float(len(values)),
        "median": float(statistics.median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(statistics.mean(values)),
    }
    if len(values) >= 2:
        result["stdev"] = float(statistics.pstdev(values))
    return result


def _print_overview(
    console: Console,
    *,
    model_key: str,
    window: WindowSelection,
    window_hours: float,
    model_runs_total: int,
    valid_window_count: int,
    computed_count: int,
) -> None:
    overview = Table(show_header=False, box=None)
    overview.add_row("[cyan]Model:[/cyan]", model_key)
    overview.add_row("[cyan]Anchor run:[/cyan]", str(window.anchor_run.get("run_name", "unknown")))
    overview.add_row("[cyan]Anchor timestamp (UTC):[/cyan]", window.anchor_timestamp.isoformat())
    overview.add_row(
        "[cyan]Window:[/cyan]",
        f"{window_hours:.4g} hour(s) ({window_hours * 60.0:.2f} minutes)",
    )
    overview.add_row("[cyan]Runs for model:[/cyan]", str(model_runs_total))
    overview.add_row("[cyan]Runs in window:[/cyan]", str(len(window.runs_in_window)))
    overview.add_row("[cyan]Valid runs in window:[/cyan]", str(valid_window_count))
    overview.add_row("[cyan]Summarized runs:[/cyan]", str(computed_count))
    if window.runs_missing_timestamp:
        overview.add_row(
            "[yellow]Missing timestamp:[/yellow]",
            str(len(window.runs_missing_timestamp)),
        )
    console.print(overview)


def _print_exclusions(console: Console, exclusions: List[Tuple[str, str]]) -> None:
    if not exclusions:
        return
    table = Table(title="Excluded Runs (Window)")
    table.add_column("Run", style="cyan")
    table.add_column("Reason", style="yellow")
    for run_name, reason in exclusions:
        table.add_row(run_name, reason)
    console.print(table)


def _format_docs_detected_cell(row: Dict[str, Any]) -> str:
    docs_detected = float(row["docs_detected"])
    docs_count = row.get("docs_detected_count")
    if isinstance(docs_count, (int, float)):
        return f"{docs_detected:.2f}% ({int(round(float(docs_count)))}/32)"
    return f"{docs_detected:.2f}%"


def _print_run_table(console: Console, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    table = Table(title="Per-Run Quality Stats (Window)")
    table.add_column("Run", style="cyan")
    table.add_column("Timestamp UTC", style="magenta")
    table.add_column("Age (min)", justify="right")
    table.add_column("Digit Top-1 (%)", justify="right")
    table.add_column("8-digit ID Top-1", justify="right")
    table.add_column("Lastname Top-1 (%)", justify="right")
    table.add_column("ID Avg d_Lev", justify="right")
    table.add_column("Lastname Avg d_Lev", justify="right")
    table.add_column("Docs detected", justify="right")
    for row in rows:
        table.add_row(
            row["run_name"],
            row["timestamp"],
            f"{row['age_minutes']:.2f}",
            f"{row['digit_top1']:.2f}",
            f"{row['id_top1']:.2f}",
            f"{row['lastname_top1']:.2f}",
            f"{row['id_avg_lev']:.4f}",
            f"{row['lastname_avg_lev']:.4f}",
            _format_docs_detected_cell(row),
        )
    console.print(table)

    cost_table = Table(title="Per-Run Cost Stats (Window)")
    cost_table.add_column("Run", style="cyan")
    cost_table.add_column("Age (min)", justify="right")
    cost_table.add_column("Total Cost ($)", justify="right")
    cost_table.add_column("Cost/Image ($)", justify="right")
    for row in rows:
        cost_table.add_row(
            row["run_name"],
            f"{row['age_minutes']:.2f}",
            f"{row['total_cost']:.6f}",
            f"{row['cost_per_image']:.8f}",
        )
    console.print(cost_table)


def _print_metric_summary(console: Console, run_rows: List[Dict[str, Any]]) -> None:
    if not run_rows:
        return
    metric_specs = [
        ("digit_top1", "Digit Top-1 (%)", 4),
        ("id_top1", "8-digit ID Top-1 (%)", 4),
        ("lastname_top1", "Lastname Top-1 (%)", 4),
        ("id_avg_lev", "ID Avg d_Lev", 6),
        ("lastname_avg_lev", "Lastname Avg d_Lev", 6),
        ("docs_detected", "Docs detected (%)", 4),
        ("cost_per_image", "Cost/Image ($)", 8),
        ("total_cost", "Total Cost ($)", 6),
    ]
    table = Table(title="Window Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("n", justify="right")
    table.add_column("median", justify="right")
    table.add_column("min", justify="right")
    table.add_column("max", justify="right")
    table.add_column("mean", justify="right")
    table.add_column("stdev", justify="right")

    for key, label, decimals in metric_specs:
        values = [float(row[key]) for row in run_rows]
        summary = summarize_values(values)
        n = int(summary.get("n", 0))
        if n == 0:
            continue
        stdev = summary.get("stdev")
        fmt = f"{{:.{decimals}f}}"
        table.add_row(
            label,
            str(n),
            fmt.format(summary["median"]),
            fmt.format(summary["min"]),
            fmt.format(summary["max"]),
            fmt.format(summary["mean"]),
            "N/A" if stdev is None else fmt.format(stdev),
        )
    console.print(table)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize recent runs for one model."
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Canonical model key: org/model[-variant] (e.g. google/gemma-3-4b-it)",
    )
    parser.add_argument("--runs-dir", default="tests/output/runs", help="Run root directory")
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Lookback window in hours from latest run (default: 24)",
    )
    parser.add_argument(
        "--window-minutes",
        type=float,
        help="Lookback window in minutes from latest run (overrides --window-hours)",
    )
    parser.add_argument(
        "--doc-info-file",
        default="imgs/q11/doc_info.csv",
        help="Document metadata CSV used for benchmark scoring",
    )
    parser.add_argument(
        "--test-ids-file",
        default="tests/data/test_ids.csv",
        help="Ground-truth IDs CSV used for benchmark scoring",
    )
    args = parser.parse_args()

    console = Console()
    try:
        window_hours = resolve_window_hours(args.window_hours, args.window_minutes)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 2

    run_manager = RunManager(args.runs_dir)
    all_runs = run_manager.list_runs()
    model_runs = filter_model_runs(all_runs, args.model)
    if not model_runs:
        console.print(f"[red]No runs found for model: {args.model}[/red]")
        return 1

    try:
        selection = select_window_runs(model_runs, window_hours)
    except ValueError as exc:
        console.print(f"[red]{exc}[/red]")
        return 1

    generator = BenchmarkTableGenerator(
        runs_base_dir=args.runs_dir,
        interactive=False,
    )

    valid_runs: List[Dict[str, Any]] = []
    exclusions: List[Tuple[str, str]] = []
    for run_info in selection.runs_in_window:
        is_valid, reason = generator._is_run_eligible_for_cohort(run_info)
        if is_valid:
            valid_runs.append(run_info)
            continue
        exclusions.append((run_info.get("run_name", "unknown"), reason or "ineligible"))

    run_rows: List[Dict[str, Any]] = []
    for run_info in valid_runs:
        run_name = run_info["run_name"]
        stats = generator.compute_run_stats(
            run_info,
            doc_info_file=args.doc_info_file,
            test_ids_file=args.test_ids_file,
        )
        if not stats:
            exclusions.append((run_name, "failed to compute stats"))
            continue

        ts = get_run_timestamp(run_info)
        if ts is None:
            exclusions.append((run_name, "missing timestamp"))
            continue

        age_minutes = (selection.anchor_timestamp - ts).total_seconds() / 60.0
        run_rows.append(
            {
                "run_name": run_name,
                "timestamp": ts.isoformat(),
                "age_minutes": age_minutes,
                "id_top1": float(stats.get("id_top1", 0.0)),
                "digit_top1": float(stats.get("digit_top1", 0.0)),
                "lastname_top1": float(stats.get("lastname_top1", 0.0)),
                "id_avg_lev": float(stats.get("id_avg_lev", 0.0)),
                "lastname_avg_lev": float(stats.get("lastname_avg_lev", 0.0)),
                "docs_detected": float(stats.get("docs_detected", 0.0)),
                "docs_detected_count": float(stats.get("docs_detected_count", 0.0)),
                "total_cost": float(stats.get("total_cost", 0.0)),
                "cost_per_image": float(stats.get("cost_per_image", 0.0)),
            }
        )

    run_rows.sort(key=lambda row: row["timestamp"], reverse=True)

    _print_overview(
        console,
        model_key=args.model,
        window=selection,
        window_hours=window_hours,
        model_runs_total=len(model_runs),
        valid_window_count=len(valid_runs),
        computed_count=len(run_rows),
    )
    _print_exclusions(console, exclusions)
    _print_run_table(console, run_rows)
    _print_metric_summary(console, run_rows)

    if not run_rows:
        console.print("[yellow]No valid runs with computed stats in the selected window.[/yellow]")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
