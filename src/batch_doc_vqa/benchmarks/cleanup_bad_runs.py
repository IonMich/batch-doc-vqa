#!/usr/bin/env python3
"""Detect and clean up invalid or incomplete benchmark run directories."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml
from rich.console import Console
from rich.table import Table


INFERENCE_FAILURE_MARKERS = {
    "_schema_failed",
    "_parse_failed",
    "_no_response",
    "_empty_response",
    "_server_error",
    "_api_error",
    "_exception",
    "_retry_failed",
}


@dataclass
class RunDiagnosis:
    run_name: str
    run_dir: Path
    model_key: Optional[str]
    timestamp: Optional[str]
    reasons: List[str]

    @property
    def is_bad(self) -> bool:
        return bool(self.reasons)


def _parse_timestamp(ts: Optional[str]) -> Optional[datetime]:
    if not ts or not isinstance(ts, str):
        return None
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _extract_timestamp(config: Optional[Dict[str, Any]], run_name: str) -> Optional[datetime]:
    if isinstance(config, dict):
        run_info = config.get("run_info")
        if isinstance(run_info, dict):
            ts = _parse_timestamp(run_info.get("timestamp"))
            if ts is not None:
                return ts

    suffix_match = re.search(r"_(\d{8}_\d{6})$", run_name)
    if suffix_match:
        return _parse_timestamp(suffix_match.group(1))
    return None


def _extract_model_key(config: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(config, dict):
        return None
    model = config.get("model")
    if not isinstance(model, dict):
        return None
    org = model.get("org")
    model_name = model.get("model")
    if not isinstance(org, str) or not isinstance(model_name, str):
        return None
    variant = model.get("variant")
    key = f"{org}/{model_name}"
    if isinstance(variant, str) and variant.strip():
        key += f"-{variant}"
    return key


def _load_yaml(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing config.yaml"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as exc:
        return None, f"invalid config.yaml ({type(exc).__name__}: {exc})"
    if not isinstance(data, dict):
        return None, "invalid config.yaml (expected object)"
    return data, None


def _load_results(path: Path) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    if not path.exists():
        return None, "missing results.json"
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        return None, f"invalid results.json ({type(exc).__name__}: {exc})"
    if not isinstance(data, dict):
        return None, "invalid results.json (expected object)"
    if not data:
        return None, "invalid results.json (empty object)"
    return data, None


def _iter_primary_entries(results: Dict[str, Any]) -> Iterable[tuple[str, Optional[Dict[str, Any]]]]:
    for image_path, raw_value in results.items():
        if isinstance(raw_value, list) and raw_value and isinstance(raw_value[0], dict):
            yield image_path, raw_value[0]
            continue
        if isinstance(raw_value, dict):
            yield image_path, raw_value
            continue
        yield image_path, None


def _entry_has_failure_marker(entry: Optional[Dict[str, Any]]) -> Optional[str]:
    if not isinstance(entry, dict):
        return "invalid entry shape"
    for marker in INFERENCE_FAILURE_MARKERS:
        value = entry.get(marker)
        if value is None:
            continue
        if isinstance(value, bool) and value is False:
            continue
        if isinstance(value, (int, float)) and value == 0:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return marker
    return None


def _count_inference_failures(results: Dict[str, Any]) -> tuple[int, Optional[str]]:
    count = 0
    sample: Optional[str] = None
    for image_path, entry in _iter_primary_entries(results):
        marker = _entry_has_failure_marker(entry)
        if marker is None:
            continue
        count += 1
        if sample is None:
            sample = f"{Path(image_path).name}: {marker}"
    return count, sample


def _count_cost_issues(results: Dict[str, Any]) -> tuple[int, int, int]:
    failed_fetches = 0
    missing_actual_cost = 0
    generation_id_leftovers = 0

    for _image_path, entry in _iter_primary_entries(results):
        if not isinstance(entry, dict):
            continue

        cost_fetch = entry.get("_cost_fetch")
        if isinstance(cost_fetch, dict):
            status = str(cost_fetch.get("status", "")).strip().lower()
            if status == "failed":
                failed_fetches += 1

        token_usage = entry.get("_token_usage")
        if not isinstance(token_usage, dict):
            missing_actual_cost += 1
            continue

        if token_usage.get("generation_id"):
            generation_id_leftovers += 1

        actual_cost = token_usage.get("actual_cost")
        if not isinstance(actual_cost, (int, float)):
            missing_actual_cost += 1

    return failed_fetches, missing_actual_cost, generation_id_leftovers


def diagnose_run(
    run_dir: Path,
    *,
    strict: bool,
    strict_costs: bool,
    strict_reproducibility: bool,
) -> RunDiagnosis:
    reasons: List[str] = []
    run_name = run_dir.name
    config_path = run_dir / "config.yaml"
    results_path = run_dir / "results.json"

    config, config_error = _load_yaml(config_path)
    if config_error:
        reasons.append(config_error)
        config = None

    model_key = _extract_model_key(config)
    run_dt = _extract_timestamp(config, run_name)
    timestamp = run_dt.strftime("%Y%m%d_%H%M%S") if run_dt else None

    if isinstance(config, dict):
        runtime = (
            config.get("environment", {}).get("runtime")
            if isinstance(config.get("environment"), dict)
            else None
        )
        if isinstance(runtime, str) and runtime.strip().upper() == "TBD":
            reasons.append("run incomplete (runtime=TBD)")

        if strict_reproducibility:
            run_info = config.get("run_info")
            if isinstance(run_info, dict):
                reproducibility = run_info.get("reproducibility")
                if isinstance(reproducibility, dict):
                    if reproducibility.get("git_dirty_relevant") is True:
                        relevant_count = reproducibility.get("git_dirty_relevant_count")
                        if isinstance(relevant_count, int) and relevant_count >= 0:
                            reasons.append(
                                f"reproducibility warning (git_dirty_relevant=true, paths={relevant_count})"
                            )
                        else:
                            reasons.append("reproducibility warning (git_dirty_relevant=true)")

    results, results_error = _load_results(results_path)
    if results_error:
        reasons.append(results_error)
        results = None

    if strict and isinstance(results, dict):
        inference_failures, sample = _count_inference_failures(results)
        if inference_failures > 0:
            suffix = f"; sample: {sample}" if sample else ""
            reasons.append(f"inference failures in {inference_failures} image(s){suffix}")

        if strict_costs:
            failed_fetches, missing_actual_cost, generation_id_leftovers = _count_cost_issues(results)
            if failed_fetches > 0:
                reasons.append(
                    f"cost fetch failed for {failed_fetches} image(s) after retries"
                )
            if generation_id_leftovers > 0:
                reasons.append(
                    f"generation_id still present for {generation_id_leftovers} image(s) (cost fetch incomplete)"
                )
            if missing_actual_cost > 0:
                reasons.append(
                    f"missing actual_cost in {missing_actual_cost} image(s)"
                )

    return RunDiagnosis(
        run_name=run_name,
        run_dir=run_dir,
        model_key=model_key,
        timestamp=timestamp,
        reasons=reasons,
    )


def _parse_yyyymmdd(value: Optional[str], *, flag: str) -> Optional[datetime.date]:
    if value is None:
        return None
    try:
        return datetime.strptime(value, "%Y%m%d").date()
    except ValueError as exc:
        raise ValueError(f"{flag} must be YYYYMMDD; got {value!r}") from exc


def _matches_filters(
    diagnosis: RunDiagnosis,
    *,
    pattern_re: Optional[re.Pattern[str]],
    model_filter: Optional[str],
    on_date: Optional[datetime.date],
    since_date: Optional[datetime.date],
    until_date: Optional[datetime.date],
) -> bool:
    if pattern_re and not pattern_re.search(diagnosis.run_name):
        return False

    if model_filter:
        if not diagnosis.model_key:
            return False
        if model_filter.lower() not in diagnosis.model_key.lower():
            return False

    run_date = None
    if diagnosis.timestamp:
        run_dt = _parse_timestamp(diagnosis.timestamp)
        run_date = run_dt.date() if run_dt else None

    if on_date and run_date != on_date:
        return False
    if since_date and (run_date is None or run_date < since_date):
        return False
    if until_date and (run_date is None or run_date > until_date):
        return False
    return True


def _safe_quarantine_destination(base_dir: Path, run_name: str) -> Path:
    candidate = base_dir / run_name
    if not candidate.exists():
        return candidate

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return base_dir / f"{run_name}_{stamp}"


def _render_table(console: Console, bad_runs: List[RunDiagnosis]) -> None:
    table = Table(title="Bad Runs", show_lines=False)
    table.add_column("Run", style="cyan")
    table.add_column("Model", style="white")
    table.add_column("Timestamp", style="magenta")
    table.add_column("Reasons", style="yellow")
    for run in bad_runs:
        model = run.model_key or "Unknown"
        ts = run.timestamp or "Unknown"
        reasons = "; ".join(run.reasons)
        table.add_row(run.run_name, model, ts, reasons)
    console.print(table)


def _write_report(report_path: Path, payload: Dict[str, Any]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Detect and optionally clean bad benchmark runs."
    )
    parser.add_argument("--runs-dir", default="tests/output/runs", help="Run root directory")
    parser.add_argument("--pattern", help="Regex filter on run directory name")
    parser.add_argument("--model", help="Substring filter on model key (e.g. google/gemma-3-4b-it)")
    parser.add_argument("--date", help="Only include runs on YYYYMMDD")
    parser.add_argument("--since", help="Only include runs on/after YYYYMMDD")
    parser.add_argument("--until", help="Only include runs on/before YYYYMMDD")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Also classify runs as bad when inference failure markers are present",
    )
    parser.add_argument(
        "--strict-costs",
        action="store_true",
        help="With --strict, also classify missing precise costs as bad",
    )
    parser.add_argument(
        "--strict-reproducibility",
        action="store_true",
        help=(
            "Classify runs as bad when run_info.reproducibility.git_dirty_relevant is true "
            "(useful for strict comparability cohorts)"
        ),
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply cleanup (default is dry-run preview only)",
    )
    parser.add_argument(
        "--quarantine-dir",
        help="Move bad runs to this directory instead of deleting",
    )
    parser.add_argument(
        "--report",
        help="Write JSON report to this file",
    )
    args = parser.parse_args()

    if args.strict_costs and not args.strict:
        parser.error("--strict-costs requires --strict")

    try:
        on_date = _parse_yyyymmdd(args.date, flag="--date")
        since_date = _parse_yyyymmdd(args.since, flag="--since")
        until_date = _parse_yyyymmdd(args.until, flag="--until")
    except ValueError as exc:
        parser.error(str(exc))

    if since_date and until_date and since_date > until_date:
        parser.error("--since must be <= --until")

    pattern_re = re.compile(args.pattern, re.IGNORECASE) if args.pattern else None
    model_filter = args.model.strip() if isinstance(args.model, str) else None
    runs_dir = Path(args.runs_dir)
    console = Console()

    if not runs_dir.exists():
        console.print(f"[red]Runs directory does not exist: {runs_dir}[/red]")
        return 1

    run_dirs = sorted([p for p in runs_dir.iterdir() if p.is_dir()], reverse=True)
    diagnosed: List[RunDiagnosis] = []
    for run_dir in run_dirs:
        diagnosis = diagnose_run(
            run_dir,
            strict=args.strict,
            strict_costs=args.strict_costs,
            strict_reproducibility=args.strict_reproducibility,
        )
        if not _matches_filters(
            diagnosis,
            pattern_re=pattern_re,
            model_filter=model_filter,
            on_date=on_date,
            since_date=since_date,
            until_date=until_date,
        ):
            continue
        diagnosed.append(diagnosis)

    bad_runs = [run for run in diagnosed if run.is_bad]

    if bad_runs:
        _render_table(console, bad_runs)
    else:
        console.print("[green]No bad runs matched the filters.[/green]")

    deleted = 0
    quarantined = 0
    failed_actions: List[Dict[str, str]] = []

    if args.apply and bad_runs:
        quarantine_base = Path(args.quarantine_dir) if args.quarantine_dir else None
        if quarantine_base:
            quarantine_base.mkdir(parents=True, exist_ok=True)

        for run in bad_runs:
            try:
                if quarantine_base:
                    destination = _safe_quarantine_destination(quarantine_base, run.run_name)
                    shutil.move(str(run.run_dir), str(destination))
                    quarantined += 1
                else:
                    shutil.rmtree(run.run_dir)
                    deleted += 1
            except Exception as exc:
                failed_actions.append(
                    {
                        "run_name": run.run_name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    elif not args.apply and bad_runs:
        console.print("[yellow]Dry-run only. Re-run with --apply to clean these runs.[/yellow]")

    summary = {
        "runs_scanned": len(run_dirs),
        "runs_matched_filters": len(diagnosed),
        "bad_runs": len(bad_runs),
        "dry_run": not args.apply,
        "deleted": deleted,
        "quarantined": quarantined,
        "action_failures": len(failed_actions),
    }
    console.print(
        f"Scanned {summary['runs_scanned']} runs, matched {summary['runs_matched_filters']}, "
        f"bad {summary['bad_runs']}, deleted {deleted}, quarantined {quarantined}."
    )
    if failed_actions:
        console.print("[red]Cleanup errors occurred for some runs.[/red]")

    if args.report:
        report_payload = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "runs_dir": str(runs_dir),
            "filters": {
                "pattern": args.pattern,
                "model": model_filter,
                "date": args.date,
                "since": args.since,
                "until": args.until,
                "strict": bool(args.strict),
                "strict_costs": bool(args.strict_costs),
                "strict_reproducibility": bool(args.strict_reproducibility),
            },
            "summary": summary,
            "bad_runs": [
                {
                    "run_name": run.run_name,
                    "run_dir": str(run.run_dir),
                    "model_key": run.model_key,
                    "timestamp": run.timestamp,
                    "reasons": run.reasons,
                }
                for run in bad_runs
            ],
            "action_failures": failed_actions,
        }
        _write_report(Path(args.report), report_payload)
        console.print(f"Wrote report to {args.report}")

    return 1 if failed_actions else 0


if __name__ == "__main__":
    raise SystemExit(main())
