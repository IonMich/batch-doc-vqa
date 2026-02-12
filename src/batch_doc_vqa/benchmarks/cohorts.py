#!/usr/bin/env python3
"""Utilities for selecting "latest cohorts" of benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class LatestCohort:
    """Represents runs grouped for one model under the latest-cohort policy."""

    model_key: str
    anchor_run: Dict[str, Any]
    runs: List[Dict[str, Any]]
    window_hours: float
    anchor_signature: Optional[Tuple[str, str]]


def extract_model_key_from_config(config: Dict[str, Any]) -> str:
    """Build canonical model key from a run config."""
    model = config.get("model", {})
    org = model.get("org", "unknown")
    model_name = model.get("model", "unknown")
    variant = model.get("variant")
    key = f"{org}/{model_name}"
    if variant:
        key += f"-{variant}"
    return key


def get_run_timestamp(run: Dict[str, Any]) -> Optional[datetime]:
    """Parse run timestamp to UTC-aware datetime."""
    config = run.get("config", {})
    run_info = config.get("run_info", {})

    timestamp_iso = run_info.get("timestamp_iso")
    if isinstance(timestamp_iso, str) and timestamp_iso.strip():
        try:
            parsed = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass

    timestamp = run_info.get("timestamp")
    if isinstance(timestamp, str) and timestamp.strip():
        try:
            parsed = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            return parsed.replace(tzinfo=timezone.utc)
        except ValueError:
            return None
    return None


def get_reproducibility_signature(run: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Return (git_commit, prompt_hash) signature when available."""
    config = run.get("config", {})
    run_info = config.get("run_info", {})
    reproducibility = run_info.get("reproducibility", {})

    git_commit = reproducibility.get("git_commit")
    prompt_hash = reproducibility.get("prompt_hash")

    if not isinstance(git_commit, str) or not git_commit or git_commit == "unknown":
        return None
    if not isinstance(prompt_hash, str) or not prompt_hash:
        return None
    return git_commit, prompt_hash


def _unique_runs_by_name(runs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    unique_runs = []
    for run in runs:
        run_name = run.get("run_name")
        if not isinstance(run_name, str):
            continue
        if run_name in seen:
            continue
        seen.add(run_name)
        unique_runs.append(run)
    return unique_runs


def select_latest_cohorts(
    runs: List[Dict[str, Any]],
    model_key_getter: Callable[[Dict[str, Any]], str] = extract_model_key_from_config,
    *,
    window_hours: float = 24.0,
) -> Dict[str, LatestCohort]:
    """Select latest cohorts per model key.

    Policy:
    - Anchor = newest run for model key.
    - Cohort members must match anchor's reproducibility signature
      (git_commit + prompt_hash) and be within `window_hours` before anchor.
    - If anchor signature is missing, fallback to single-run cohort (anchor only).
    """

    deduped_runs = _unique_runs_by_name(runs)
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for run in deduped_runs:
        config = run.get("config", {})
        if not isinstance(config, dict):
            continue
        model_key = model_key_getter(config)
        grouped.setdefault(model_key, []).append(run)

    def _sort_key(run: Dict[str, Any]) -> datetime:
        timestamp = get_run_timestamp(run)
        if timestamp is not None:
            return timestamp
        return datetime.min.replace(tzinfo=timezone.utc)

    cohorts_by_model: Dict[str, LatestCohort] = {}
    for model_key, model_runs in grouped.items():
        model_runs.sort(key=_sort_key, reverse=True)
        anchor = model_runs[0]
        anchor_timestamp = get_run_timestamp(anchor)
        anchor_signature = get_reproducibility_signature(anchor)

        cohort_runs = [anchor]
        if anchor_signature is not None and anchor_timestamp is not None and window_hours > 0:
            max_age = timedelta(hours=window_hours)
            for candidate in model_runs[1:]:
                candidate_signature = get_reproducibility_signature(candidate)
                if candidate_signature != anchor_signature:
                    continue

                candidate_timestamp = get_run_timestamp(candidate)
                if candidate_timestamp is None:
                    continue

                age = anchor_timestamp - candidate_timestamp
                if age.total_seconds() < 0:
                    continue
                if age <= max_age:
                    cohort_runs.append(candidate)

        cohorts_by_model[model_key] = LatestCohort(
            model_key=model_key,
            anchor_run=anchor,
            runs=cohort_runs,
            window_hours=window_hours,
            anchor_signature=anchor_signature,
        )

    # Keep deterministic model ordering by latest anchor timestamp (newest first).
    ordered_items = sorted(
        cohorts_by_model.items(),
        key=lambda item: _sort_key(item[1].anchor_run),
        reverse=True,
    )
    return dict(ordered_items)


def format_cohort_debug_report(cohorts: Dict[str, LatestCohort]) -> str:
    """Render human-readable cohort grouping details."""
    if not cohorts:
        return "No cohorts selected."

    lines: List[str] = []
    lines.append("Latest cohort selection report:")
    for model_key, cohort in cohorts.items():
        anchor = cohort.anchor_run.get("run_name", "unknown")
        sig = cohort.anchor_signature
        if sig is None:
            sig_text = "signature=missing (fallback to anchor only)"
        else:
            commit_short = sig[0][:8]
            prompt_short = sig[1][:10]
            sig_text = f"signature=({commit_short}, {prompt_short}...)"
        lines.append(
            f"- {model_key}: n={len(cohort.runs)}, anchor={anchor}, {sig_text}"
        )
        for run in cohort.runs:
            run_name = run.get("run_name", "unknown")
            ts = get_run_timestamp(run)
            ts_text = ts.isoformat() if ts else "unknown-ts"
            lines.append(f"  - {run_name} @ {ts_text}")
    return "\n".join(lines)
