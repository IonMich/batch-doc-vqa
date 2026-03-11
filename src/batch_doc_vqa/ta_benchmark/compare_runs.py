#!/usr/bin/env python3
"""Compare multiple TA score outputs and optionally generate a Pareto plot."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.table import Table

from .io_utils import read_json


console = Console()


@dataclass(frozen=True)
class ScoreRow:
    source_file: str
    model: str
    run: str
    metric: float
    cost: float
    runtime_seconds: Optional[float]
    total_cost: Optional[float]


def _resolve_metric(payload: dict[str, Any], path: str) -> Optional[float]:
    cur: Any = payload
    for token in path.split("."):
        if not isinstance(cur, dict) or token not in cur:
            return None
        cur = cur[token]
    try:
        return float(cur)
    except (TypeError, ValueError):
        return None


def _discover_files(scores_glob: Optional[str], scores_csv: Optional[str]) -> list[Path]:
    files: list[Path] = []
    if scores_glob:
        files.extend(sorted(Path().glob(scores_glob)))
    if scores_csv:
        for chunk in scores_csv.split(","):
            text = chunk.strip()
            if text:
                files.append(Path(text))
    unique = sorted({path.expanduser().resolve(strict=False) for path in files if path.exists()})
    return unique


def _load_rows(files: list[Path], *, metric_path: str, cost_path: str) -> list[ScoreRow]:
    rows: list[ScoreRow] = []
    for path in files:
        payload = read_json(path)
        if not isinstance(payload, dict):
            continue
        metric_value = _resolve_metric(payload, f"scores.{metric_path}")
        cost_value = _resolve_metric(payload, cost_path)
        if metric_value is None or cost_value is None:
            continue
        run_meta = payload.get("run_metadata", {}) if isinstance(payload.get("run_metadata"), dict) else {}
        model = str(run_meta.get("model", "") or "").strip() or path.stem
        run = str(run_meta.get("source_run", "") or "").strip() or path.stem
        runtime = _resolve_metric(payload, "run_metadata.runtime_seconds")
        total_cost = _resolve_metric(payload, "run_metadata.total_cost")
        rows.append(
            ScoreRow(
                source_file=str(path),
                model=model,
                run=run,
                metric=float(metric_value),
                cost=float(cost_value),
                runtime_seconds=runtime,
                total_cost=total_cost,
            )
        )
    return rows


def _pareto_frontier(rows: list[ScoreRow], *, higher_is_better: bool) -> list[ScoreRow]:
    ordered = sorted(rows, key=lambda row: (row.cost, -row.metric if higher_is_better else row.metric))
    frontier: list[ScoreRow] = []
    best_metric: Optional[float] = None
    for row in ordered:
        if best_metric is None:
            frontier.append(row)
            best_metric = row.metric
            continue
        if higher_is_better:
            if row.metric > best_metric:
                frontier.append(row)
                best_metric = row.metric
        else:
            if row.metric < best_metric:
                frontier.append(row)
                best_metric = row.metric
    return frontier


def _write_markdown(
    *,
    output_md: Path,
    rows: list[ScoreRow],
    frontier: list[ScoreRow],
    metric_path: str,
    cost_path: str,
) -> None:
    frontier_runs = {row.run for row in frontier}
    lines: list[str] = []
    lines.append("# TA Run Comparison")
    lines.append("")
    lines.append(f"- Metric: `{metric_path}`")
    lines.append(f"- Cost field: `{cost_path}`")
    lines.append("")
    lines.append("| Model | Run | Metric | Cost | Runtime (s) | Total Cost | Pareto | Source |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | --- | --- |")
    for row in sorted(rows, key=lambda item: (-item.metric, item.cost, item.model)):
        lines.append(
            "| "
            f"{row.model} | {row.run} | {row.metric:.6f} | {row.cost:.8f} | "
            f"{'' if row.runtime_seconds is None else f'{row.runtime_seconds:.3f}'} | "
            f"{'' if row.total_cost is None else f'{row.total_cost:.6f}'} | "
            f"{'yes' if row.run in frontier_runs else 'no'} | `{row.source_file}` |"
        )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _plot_pareto(
    *,
    output_path: Path,
    rows: list[ScoreRow],
    frontier: list[ScoreRow],
    title: str,
    metric_label: str,
    higher_is_better: bool,
) -> None:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    plt.figure(figsize=(10, 6))
    plt.scatter([r.cost for r in rows], [r.metric for r in rows], alpha=0.65, label="Runs")

    frontier_sorted = sorted(frontier, key=lambda item: item.cost)
    plt.plot(
        [r.cost for r in frontier_sorted],
        [r.metric for r in frontier_sorted],
        color="red",
        linewidth=2,
        marker="o",
        label="Pareto frontier",
    )
    for row in frontier_sorted:
        plt.annotate(row.model, (row.cost, row.metric), fontsize=8, xytext=(4, 4), textcoords="offset points")

    plt.xlabel("Cost per image")
    plt.ylabel(metric_label)
    plt.title(title)
    if not higher_is_better:
        plt.gca().invert_yaxis()
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare TA score runs and compute Pareto frontier.")
    parser.add_argument("--scores-glob", help="Glob pattern for score JSON files")
    parser.add_argument("--scores", help="Comma-separated explicit score JSON file paths")
    parser.add_argument(
        "--metric",
        default="rubric_scoring.qwk",
        help="Metric path under scores.* (default: rubric_scoring.qwk)",
    )
    parser.add_argument(
        "--cost-field",
        default="run_metadata.cost_per_image",
        help="Cost field path (default: run_metadata.cost_per_image)",
    )
    parser.add_argument(
        "--higher-is-better",
        action="store_true",
        default=True,
        help="Metric direction: higher values are better (default: true)",
    )
    parser.add_argument(
        "--lower-is-better",
        action="store_true",
        help="Metric direction: lower values are better",
    )
    parser.add_argument("--output-md", required=True, help="Output markdown comparison table")
    parser.add_argument("--pareto-output", help="Optional output PNG for Pareto plot")
    parser.add_argument("--title", default="TA Benchmark Pareto Frontier", help="Plot title")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    higher_is_better = not bool(args.lower_is_better)

    files = _discover_files(args.scores_glob, args.scores)
    if not files:
        console.print("[red]No score files found.[/red]")
        return 2

    rows = _load_rows(files, metric_path=args.metric, cost_path=args.cost_field)
    if not rows:
        console.print("[red]No rows with both metric and cost values were found.[/red]")
        return 2
    frontier = _pareto_frontier(rows, higher_is_better=higher_is_better)

    output_md = Path(args.output_md).expanduser().resolve(strict=False)
    _write_markdown(
        output_md=output_md,
        rows=rows,
        frontier=frontier,
        metric_path=args.metric,
        cost_path=args.cost_field,
    )

    if args.pareto_output:
        plot_path = Path(args.pareto_output).expanduser().resolve(strict=False)
        try:
            _plot_pareto(
                output_path=plot_path,
                rows=rows,
                frontier=frontier,
                title=args.title,
                metric_label=args.metric,
                higher_is_better=higher_is_better,
            )
            plot_note = str(plot_path)
        except Exception as exc:
            plot_note = f"failed ({type(exc).__name__}: {exc})"
    else:
        plot_note = "N/A"

    table = Table(title="TA Run Comparison")
    table.add_column("Field", style="cyan")
    table.add_column("Value")
    table.add_row("Rows", str(len(rows)))
    table.add_row("Pareto points", str(len(frontier)))
    table.add_row("Output markdown", str(output_md))
    table.add_row("Pareto plot", plot_note)
    console.print(table)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
