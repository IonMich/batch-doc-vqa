"""Sanitized, versioned benchmark-run summaries for publication.

Raw inference runs intentionally remain machine-local.  This module exports the
small, auditable subset needed to regenerate the public benchmark tables and
plots without copying model responses, identifiers, absolute paths, or API
metadata into Git.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


PUBLISHED_RUN_SCHEMA_VERSION = 1
PUBLISHED_ARCHIVE_SCHEMA_VERSION = 1
DEFAULT_PUBLISHED_ROOT = Path("benchmarks/published/q11")
DEFAULT_PUBLISHED_RUNS_DIR = DEFAULT_PUBLISHED_ROOT / "runs"
DEFAULT_ARCHIVE_MANIFEST = DEFAULT_PUBLISHED_ROOT / "archive.json"

_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s\"'])/(?:Users|home)/[^\s\"']+")
_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|password|secret|access[_-]?token|refresh[_-]?token)",
    re.IGNORECASE,
)

_SAFE_MODEL_KEYS = ("org", "model", "variant", "model_size", "open_weights", "license_info")
_SAFE_API_KEYS = (
    "provider",
    "temperature",
    "max_tokens",
    "top_p",
    "top_k",
    "min_p",
    "presence_penalty",
    "repetition_penalty",
)
_SAFE_FEATURE_KEYS = ("structured_output", "regex_patterns")
_SAFE_ENVIRONMENT_KEYS = ("runtime",)
_SAFE_REPRODUCIBILITY_KEYS = (
    "manifest_version",
    "git_commit",
    "prompt_hash",
    "inference_settings_hash",
    "parser_version",
    "schema_version",
)
_SAFE_ADDITIONAL_KEYS = (
    "actual_model_providers",
    "concurrency",
    "cost_fetch_max_workers",
    "dataset_manifest_sha256",
    "endpoint_type",
    "extraction_mode",
    "generation_param_sources",
    "generation_params_effective",
    "images_dir",
    "model_context_length",
    "model_name",
    "model_pricing",
    "pages",
    "preset_id",
    "provider_routing_effective",
    "provider_routing_requested",
    "rate_limit",
    "response_format",
    "retry_base_delay",
    "retry_max",
    "schema_hash",
    "schema_retry_max",
    "strict_schema",
    "token_escalation_repetition_threshold",
)
_PUBLISHED_STATS_KEYS = (
    "digit_top1",
    "id_top1",
    "lastname_top1",
    "id_avg_lev",
    "lastname_avg_lev",
    "docs_detected",
    "docs_detected_count",
    "expected_docs_count",
    "runtime_seconds",
    "fully_parallelizable_runtime_available",
    "fully_parallelizable_runtime_seconds",
    "timed_images",
    "total_images",
    "cost_per_image",
    "total_cost",
    "observed_total_cost",
    "cost_status",
    "cost_complete",
    "total_requests",
    "costed_requests",
    "precise_cost_requests",
    "estimated_cost_requests",
    "missing_cost_requests",
    "zero_cost_precise_requests",
)
_PREDICTION_FIELDS = ("student_full_name", "university_id", "ufid", "section_number")
_FAILURE_MARKERS = (
    "_schema_failed",
    "_parse_failed",
    "_no_response",
    "_empty_response",
    "_server_error",
    "_api_error",
    "_exception",
    "_retry_failed",
)


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any) -> str:
    """Return a SHA-256 hash of a canonical JSON representation."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: str | Path) -> str:
    """Return a content hash without exposing the source path in the result."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_dataset_provenance(
    doc_info_file: str | Path,
    test_ids_file: str | Path,
    *,
    dataset_id: str = "q11",
) -> Dict[str, Any]:
    """Create a stable dataset identity from the scoring inputs' contents."""
    doc_info_path = Path(doc_info_file)
    test_ids_path = Path(test_ids_file)
    with doc_info_path.open(newline="", encoding="utf-8") as source:
        doc_rows = list(csv.DictReader(source))
    with test_ids_path.open(newline="", encoding="utf-8") as source:
        test_rows = list(csv.DictReader(source))

    docs = {str(row.get("doc", "")) for row in test_rows if str(row.get("doc", ""))}
    pages = sorted(
        {
            int(str(row["page"]))
            for row in doc_rows
            if str(row.get("page", "")).strip().isdigit()
        }
    )
    provenance = {
        "dataset_id": dataset_id,
        "doc_info_sha256": file_hash(doc_info_path),
        "test_ids_sha256": file_hash(test_ids_path),
        "expected_docs": len(docs) or len(test_rows),
        "manifest_rows": len(doc_rows),
        "pages": pages,
    }
    # The human-readable dataset ID is metadata, not content.  Equivalent
    # inputs must have the same identity even if an operator gives them a
    # different local label.
    provenance["content_hash"] = stable_hash({key: value for key, value in provenance.items() if key != "dataset_id"})
    return provenance


def _safe_subset(source: Mapping[str, Any], keys: Iterable[str]) -> Dict[str, Any]:
    return {key: source[key] for key in keys if key in source}


def _contains_forbidden_value(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(
            _SENSITIVE_KEY_RE.search(str(key)) or _contains_forbidden_value(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return any(_contains_forbidden_value(child) for child in value)
    return isinstance(value, str) and bool(_ABSOLUTE_PATH_RE.search(value))


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(child) for key, child in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sanitize_config(config: Mapping[str, Any]) -> Dict[str, Any]:
    """Keep only safe, reproducibility-relevant configuration fields."""
    run_info = config.get("run_info", {}) if isinstance(config.get("run_info"), Mapping) else {}
    reproducibility = run_info.get("reproducibility", {}) if isinstance(run_info.get("reproducibility"), Mapping) else {}
    additional = config.get("additional", {}) if isinstance(config.get("additional"), Mapping) else {}

    sanitized = {
        "run_info": {
            "run_name": run_info.get("run_name"),
            "timestamp": run_info.get("timestamp"),
            "timestamp_iso": run_info.get("timestamp_iso"),
            "reproducibility": _safe_subset(reproducibility, _SAFE_REPRODUCIBILITY_KEYS),
        },
        "model": _safe_subset(config.get("model", {}), _SAFE_MODEL_KEYS),
        "api": _safe_subset(config.get("api", {}), _SAFE_API_KEYS),
        "features": _safe_subset(config.get("features", {}), _SAFE_FEATURE_KEYS),
        "environment": _safe_subset(config.get("environment", {}), _SAFE_ENVIRONMENT_KEYS),
        "additional": _safe_subset(additional, _SAFE_ADDITIONAL_KEYS),
    }
    sanitized = _json_safe(sanitized)
    if _contains_forbidden_value(sanitized):
        raise ValueError("sanitized configuration still contains a secret-like key or absolute local path")
    return sanitized


def build_aggregation_inputs(config: Mapping[str, Any], dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Return all fields that must match before runs are aggregated."""
    safe = sanitize_config(config)
    reproducibility = safe["run_info"].get("reproducibility", {})
    additional = safe.get("additional", {})
    return {
        "version": 1,
        "dataset_content_hash": dataset.get("content_hash"),
        "dataset_pages": dataset.get("pages", []),
        "run_manifest_sha256": safe.get("additional", {}).get("dataset_manifest_sha256"),
        "model": safe.get("model", {}),
        "api": safe.get("api", {}),
        "features": safe.get("features", {}),
        "generation_params": additional.get("generation_params_effective", {}),
        "generation_param_sources": additional.get("generation_param_sources", {}),
        "schema_hash": additional.get("schema_hash"),
        "strict_schema": additional.get("strict_schema"),
        "extraction_mode": additional.get("extraction_mode"),
        "response_format": additional.get("response_format"),
        "routing": {
            "requested": additional.get("provider_routing_requested", {}),
            "effective": additional.get("provider_routing_effective", {}),
            "actual_model_providers": additional.get("actual_model_providers", []),
        },
        "runtime_settings": {
            key: additional.get(key)
            for key in ("concurrency", "rate_limit", "retry_base_delay", "retry_max", "schema_retry_max")
            if key in additional
        },
        "reproducibility": {
            key: reproducibility.get(key)
            for key in ("git_commit", "prompt_hash", "inference_settings_hash", "parser_version", "schema_version")
            if key in reproducibility
        },
    }


def build_aggregation_fingerprint(config: Mapping[str, Any], dataset: Mapping[str, Any]) -> str:
    """Return the explicit, content-addressed compatibility fingerprint."""
    return stable_hash(build_aggregation_inputs(config, dataset))


def build_sanitized_request_evidence(raw_results: Mapping[str, Any]) -> Dict[str, Any]:
    """Retain per-request operational evidence without retaining predictions.

    Raw names, IDs, prompts, image paths, response bodies, and provider IDs are
    deliberately excluded.  The resulting records are still rich enough for
    latency/cost distributions, schema/failure analysis, field-completeness
    studies, and additional aggregate benchmark byproducts.
    """
    records: List[Dict[str, Any]] = []
    for index, source_key in enumerate(sorted(raw_results)):
        entries = raw_results.get(source_key)
        entry = entries[0] if isinstance(entries, list) and entries and isinstance(entries[0], Mapping) else {}
        token_usage = entry.get("_token_usage", {}) if isinstance(entry.get("_token_usage"), Mapping) else {}
        timing = entry.get("_timing", {}) if isinstance(entry.get("_timing"), Mapping) else {}
        generation = entry.get("_generation_meta", {}) if isinstance(entry.get("_generation_meta"), Mapping) else {}
        response_fields = {
            field: {
                "present": field in entry and entry.get(field) not in (None, ""),
                "character_count": len(entry[field]) if isinstance(entry.get(field), str) else None,
            }
            for field in _PREDICTION_FIELDS
        }
        failure_markers = [
            marker
            for marker in _FAILURE_MARKERS
            if entry.get(marker) not in (None, False, 0, "")
        ]
        actual_cost = token_usage.get("actual_cost")
        records.append(
            {
                "request_index": index,
                # This is a one-way identifier for joining records within the
                # public archive; the original image path is never exported.
                "document_ref": stable_hash({"source_key": str(source_key)}),
                "response_fields": response_fields,
                "failure_markers": failure_markers,
                "timing": {
                    "elapsed_seconds": timing.get("elapsed_seconds"),
                },
                "token_usage": {
                    "prompt_tokens": token_usage.get("prompt_tokens"),
                    "completion_tokens": token_usage.get("completion_tokens"),
                    "total_tokens": token_usage.get("total_tokens"),
                },
                "cost": {
                    "actual_cost": actual_cost if isinstance(actual_cost, (int, float)) else None,
                    "provenance": "precise" if isinstance(actual_cost, (int, float)) and actual_cost >= 0 else "not-precise",
                },
                "generation": {
                    "model": generation.get("model"),
                    "finish_reason": generation.get("finish_reason"),
                    "native_finish_reason": generation.get("native_finish_reason"),
                    "service_tier": generation.get("service_tier"),
                    "latency": generation.get("latency"),
                },
            }
        )
    return {"schema_version": 1, "request_count": len(records), "records": records}


def make_published_run_summary(
    run_info: Mapping[str, Any],
    stats: Mapping[str, Any],
    dataset: Mapping[str, Any],
    *,
    request_evidence: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a redacted published-run summary from one local raw run."""
    config = run_info.get("config", {})
    if not isinstance(config, Mapping):
        raise ValueError("run config is missing or invalid")
    safe_config = sanitize_config(config)
    run_name = str(run_info.get("run_name") or safe_config["run_info"].get("run_name") or "")
    if not run_name:
        raise ValueError("run name is required")
    published_stats = {key: _json_safe(stats[key]) for key in _PUBLISHED_STATS_KEYS if key in stats}
    aggregation_inputs = build_aggregation_inputs(config, dataset)
    summary = {
        "schema_version": PUBLISHED_RUN_SCHEMA_VERSION,
        "run_name": run_name,
        "dataset": _json_safe(dict(dataset)),
        "config": safe_config,
        "aggregation_fingerprint": stable_hash(aggregation_inputs),
        "aggregation_inputs": aggregation_inputs,
        "stats": published_stats,
    }
    if request_evidence is not None:
        summary["request_evidence"] = _json_safe(dict(request_evidence))
    validate_published_summary(summary)
    return summary


def validate_published_summary(summary: Mapping[str, Any]) -> None:
    """Reject malformed or unsafe published summaries before writing/reading them."""
    if summary.get("schema_version") != PUBLISHED_RUN_SCHEMA_VERSION:
        raise ValueError("unsupported published run schema version")
    if not isinstance(summary.get("run_name"), str) or not summary["run_name"].strip():
        raise ValueError("published run summary is missing run_name")
    dataset = summary.get("dataset")
    if not isinstance(dataset, Mapping) or not isinstance(dataset.get("content_hash"), str):
        raise ValueError("published run summary is missing dataset content hash")
    if not isinstance(summary.get("config"), Mapping):
        raise ValueError("published run summary is missing config")
    if not isinstance(summary.get("aggregation_fingerprint"), str):
        raise ValueError("published run summary is missing aggregation fingerprint")
    if not isinstance(summary.get("stats"), Mapping):
        raise ValueError("published run summary is missing stats")
    if _contains_forbidden_value(summary):
        raise ValueError("published run summary contains a secret-like key or absolute local path")


def published_summary_path(published_runs_dir: str | Path, run_name: str) -> Path:
    """Return a safe deterministic output filename for a published run."""
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", run_name).strip(".-")
    if not safe_name:
        raise ValueError("run name cannot produce an empty published filename")
    return Path(published_runs_dir) / f"{safe_name}.json"


def write_published_summary(path: str | Path, summary: Mapping[str, Any]) -> None:
    validate_published_summary(summary)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_published_summaries(published_runs_dir: str | Path) -> List[Dict[str, Any]]:
    """Load validated published summaries in deterministic order."""
    root = Path(published_runs_dir)
    if not root.exists():
        return []
    summaries: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        validate_published_summary(payload)
        summaries.append(payload)
    return summaries


def summary_to_run_info(summary: Mapping[str, Any]) -> Dict[str, Any]:
    """Adapt a published summary to the table generator's run-info shape."""
    validate_published_summary(summary)
    config = dict(summary["config"])
    run_info = dict(config.get("run_info", {}))
    reproducibility = dict(run_info.get("reproducibility", {}))
    reproducibility["aggregation_fingerprint"] = summary["aggregation_fingerprint"]
    run_info["reproducibility"] = reproducibility
    config["run_info"] = run_info
    return {
        "run_name": summary["run_name"],
        "config": config,
        "has_results": True,
        "has_table_results": True,
        "has_manifest": True,
        "published_summary": dict(summary),
        "dataset_content_hash": summary["dataset"]["content_hash"],
    }


def build_archive_manifest(summaries: Sequence[Mapping[str, Any]], dataset: Mapping[str, Any]) -> Dict[str, Any]:
    """Build an explicit completeness marker for a published archive."""
    run_entries = []
    for summary in sorted(summaries, key=lambda item: str(item["run_name"])):
        validate_published_summary(summary)
        if summary["dataset"].get("content_hash") != dataset.get("content_hash"):
            raise ValueError("cannot finalize an archive containing multiple dataset content hashes")
        run_entries.append(
            {
                "run_name": summary["run_name"],
                "aggregation_fingerprint": summary["aggregation_fingerprint"],
                "summary_sha256": stable_hash(summary),
            }
        )
    manifest = {
        "schema_version": PUBLISHED_ARCHIVE_SCHEMA_VERSION,
        "publication_status": "complete",
        "dataset": _json_safe(dict(dataset)),
        "runs": run_entries,
    }
    manifest["archive_sha256"] = stable_hash(manifest)
    return manifest


def is_complete_published_archive(
    published_runs_dir: str | Path = DEFAULT_PUBLISHED_RUNS_DIR,
    archive_manifest: str | Path = DEFAULT_ARCHIVE_MANIFEST,
) -> bool:
    """Return whether the archive has been explicitly finalized and validates."""
    manifest_path = Path(archive_manifest)
    if not manifest_path.exists():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        summaries = load_published_summaries(published_runs_dir)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    if manifest.get("schema_version") != PUBLISHED_ARCHIVE_SCHEMA_VERSION:
        return False
    if manifest.get("publication_status") != "complete":
        return False
    try:
        expected = build_archive_manifest(summaries, manifest["dataset"])
    except (KeyError, ValueError):
        return False
    return manifest == expected


def _matching_runs(generator: Any, patterns: Optional[Sequence[str]]) -> List[Dict[str, Any]]:
    return generator._collect_matching_runs(list(patterns) if patterns else None)


def publish_local_runs(
    *,
    runs_dir: str | Path,
    published_runs_dir: str | Path,
    doc_info_file: str | Path,
    test_ids_file: str | Path,
    dataset_id: str,
    patterns: Optional[Sequence[str]] = None,
    check: bool = False,
    finalize: bool = False,
) -> List[Path]:
    """Validate local raw runs and export their sanitized public summaries."""
    # Imported lazily to avoid an import cycle: table_generator also consumes
    # the published summaries.
    from .table_generator import BenchmarkTableGenerator

    generator = BenchmarkTableGenerator(
        str(runs_dir),
        interactive=False,
        source="local",
        cache_results=not check,
        write_metadata=not check,
    )
    dataset = build_dataset_provenance(doc_info_file, test_ids_file, dataset_id=dataset_id)
    output_paths: List[Path] = []
    summaries: List[Dict[str, Any]] = []
    failures: List[str] = []
    for run_info in _matching_runs(generator, patterns):
        dataset_ok, reason = generator._matches_requested_dataset(
            run_info,
            doc_info_file=str(doc_info_file),
            test_ids_file=str(test_ids_file),
            dataset_provenance=dataset,
            validate_result_paths=True,
        )
        if not dataset_ok:
            failures.append(f"{run_info.get('run_name', 'unknown')}: {reason or 'dataset mismatch'}")
            continue
        eligible, reason = generator._is_run_eligible_for_cohort(run_info)
        if not eligible:
            failures.append(f"{run_info.get('run_name', 'unknown')}: {reason or 'ineligible'}")
            continue
        stats = generator.compute_run_stats(run_info, str(doc_info_file), str(test_ids_file), dataset_provenance=dataset)
        if not stats:
            failures.append(f"{run_info.get('run_name', 'unknown')}: unable to compute stats")
            continue
        raw_results = generator.run_manager.load_results(run_info["run_name"])
        summary = make_published_run_summary(
            run_info,
            stats,
            dataset,
            request_evidence=build_sanitized_request_evidence(raw_results),
        )
        destination = published_summary_path(published_runs_dir, summary["run_name"])
        serialized = json.dumps(summary, indent=2, sort_keys=True) + "\n"
        if check:
            if not destination.exists() or destination.read_text(encoding="utf-8") != serialized:
                failures.append(f"{summary['run_name']}: published summary is missing or stale")
        else:
            write_published_summary(destination, summary)
        output_paths.append(destination)
        summaries.append(summary)

    if failures:
        raise RuntimeError("Publish validation failed:\n" + "\n".join(failures))
    if not summaries:
        raise RuntimeError("No eligible runs were available to publish")

    if finalize:
        all_summaries = load_published_summaries(published_runs_dir)
        manifest = build_archive_manifest(all_summaries, dataset)
        manifest_path = Path(published_runs_dir).parent / "archive.json"
        serialized = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        if check:
            if not manifest_path.exists() or manifest_path.read_text(encoding="utf-8") != serialized:
                raise RuntimeError("Publish validation failed:\narchive manifest is missing or stale")
        else:
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(serialized, encoding="utf-8")
        output_paths.append(manifest_path)
    return output_paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish sanitized benchmark summaries from local raw runs")
    parser.add_argument("--runs-dir", default="tests/output/runs", help="Machine-local raw-run directory")
    parser.add_argument("--published-runs-dir", default=str(DEFAULT_PUBLISHED_RUNS_DIR), help="Tracked published summary directory")
    parser.add_argument("--doc-info", default="imgs/q11/doc_info.csv", help="Scoring document manifest")
    parser.add_argument("--test-ids", default="tests/data/test_ids.csv", help="Scoring ground-truth CSV")
    parser.add_argument("--dataset-id", default="q11", help="Stable public dataset identifier")
    parser.add_argument("--patterns", nargs="*", help="Optional run-name regex filters")
    parser.add_argument("--check", action="store_true", help="Validate that the published archive is current without writing")
    parser.add_argument("--finalize", action="store_true", help="Write or verify the complete archive manifest")
    args = parser.parse_args()
    paths = publish_local_runs(
        runs_dir=args.runs_dir,
        published_runs_dir=args.published_runs_dir,
        doc_info_file=args.doc_info,
        test_ids_file=args.test_ids,
        dataset_id=args.dataset_id,
        patterns=args.patterns,
        check=args.check,
        finalize=args.finalize,
    )
    mode = "Validated" if args.check else "Published"
    print(f"{mode} {len(paths)} artifact(s).")


if __name__ == "__main__":
    main()
