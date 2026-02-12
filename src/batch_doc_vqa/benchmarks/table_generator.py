#!/usr/bin/env python3
"""
Enhanced benchmark table generation using the new run management system.
Automatically discovers and processes runs from tests/output/runs.
"""
import os
import pandas as pd
import argparse
import json
import random
import statistics
from typing import Dict, List, Optional, Any, Tuple

from rich.console import Console
from rich.table import Table

from ..core.run_manager import RunManager
from ..core import format_runtime
from .cohorts import (
    select_latest_cohorts,
    format_cohort_debug_report,
)
from ..utils.string_matching import (
    get_llm_ids_and_fullnames,
    get_llm_distances,
    get_matches,
)


class BenchmarkTableGenerator:
    """Enhanced benchmark table generator using run management system."""
    
    # Default rows to include in tables
    DEFAULT_ROWS = [
        "LLM model size",
        "Open-weights", 
        "digit_top1",
        "8-digit id_top1",
        "lastname_top1",
        "ID Avg d_Lev",
        "Lastname Avg d_Lev",
        "Docs detected",
        "Runtime",
        "Cost per image",
        "Total cost"
    ]

    # Any of these markers indicates inference did not fully succeed for that image.
    # Runs containing one or more such failures are excluded from latest-cohort aggregation.
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
    
    # Hardcoded OpenCV+CNN baseline data
    OPENCV_CNN_BASELINE = {
        "LLM model size": "N/A",
        "Open-weights": "N/A",
        "digit_top1": "85.16%",
        "8-digit id_top1": "??",
        "lastname_top1": "N/A", 
        "ID Avg d_Lev": "N/A",
        "Lastname Avg d_Lev": "N/A",
        "Docs detected": "90.62% (29/32)",
        "Runtime": "~1 second",
        "Cost per image": "$0.00",
        "Total cost": "$0.00"
    }
    
    def __init__(self, runs_base_dir: str = "tests/output/runs", metadata_file: str = "model_metadata.json", 
                 interactive: bool = True, included_rows: List[str] = None):
        self.run_manager = RunManager(runs_base_dir)
        self.console = Console()
        self.metadata_file = metadata_file
        self.interactive = interactive
        self.included_rows = included_rows or self.DEFAULT_ROWS
        self.model_metadata = self._load_model_metadata()
        
    def _load_model_metadata(self) -> Dict:
        """Load model metadata from JSON file."""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                print(f"Warning: Could not load {self.metadata_file}, creating new metadata")
        
        # Return default structure if file doesn't exist or is invalid
        return {"models": {}, "needs_review": []}
    
    def _save_model_metadata(self):
        """Save model metadata to JSON file."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.model_metadata, f, indent=2)
    
    def _get_model_key(self, config: Dict) -> str:
        """Generate model key from config."""
        model_key = f"{config['model']['org']}/{config['model']['model']}"
        if config['model']['variant']:
            model_key += f"-{config['model']['variant']}"
        return model_key
    
    def _review_unknown_models(self, unknown_models: List[str]):
        """Interactively review unknown models."""
        if not unknown_models:
            return
            
        print(f"\n🔍 Found {len(unknown_models)} unknown model(s) that need metadata review:")
        for model in unknown_models:
            print(f"  - {model}")
        
        if not self.interactive:
            print("Non-interactive mode: Models added to needs_review list.")
            for model in unknown_models:
                if model not in self.model_metadata["needs_review"]:
                    self.model_metadata["needs_review"].append(model)
            self._save_model_metadata()
            return
        
        try:
            proceed = input("\nWould you like to review these models now? (y/n): ").lower().strip()
        except EOFError:
            print("Non-interactive environment detected. Models added to needs_review list.")
            for model in unknown_models:
                if model not in self.model_metadata["needs_review"]:
                    self.model_metadata["needs_review"].append(model)
            self._save_model_metadata()
            return
            
        if proceed != 'y':
            print("Skipping review. Models added to needs_review list.")
            for model in unknown_models:
                if model not in self.model_metadata["needs_review"]:
                    self.model_metadata["needs_review"].append(model)
            self._save_model_metadata()
            return
        
        for model in unknown_models:
            print(f"\n📝 Reviewing model: {model}")
            
            # Ask if open-weights
            while True:
                open_weights = input("Is this model open-weights? (y/n): ").lower().strip()
                if open_weights in ['y', 'yes']:
                    is_open = True
                    break
                elif open_weights in ['n', 'no']:
                    is_open = False
                    break
                else:
                    print("Please enter 'y' or 'n'")
            
            # Ask for model size (only if open-weights)
            model_size = "??"
            if is_open:
                while True:
                    model_size = input("Number of parameters (e.g., 8B, 120A4, ??): ").strip()
                    if model_size:
                        break
                    else:
                        print("Please enter a model size")
            
            # Ask for license (only if open-weights)
            license_info = None
            if is_open:
                license_info = input("Most permissive license offered (e.g., Apache-2.0, MIT, custom): ").strip()
                if not license_info:
                    license_info = "unknown"
            
            # Add to metadata
            self.model_metadata["models"][model] = {
                "open_weights": is_open,
                "model_size": model_size,
                "license": license_info if is_open else None
            }
            
            # Remove from needs_review if it was there
            if model in self.model_metadata["needs_review"]:
                self.model_metadata["needs_review"].remove(model)
            
            # Save immediately after each model to avoid losing progress
            self._save_model_metadata()
            print(f"✅ Added metadata for {model} (saved)")
        
        print("\n💾 All model metadata saved!")
    
    def compute_run_stats(self, run_info: Dict, doc_info_file: str, test_ids_file: str) -> Optional[Dict]:
        """Compute statistics for a single run."""
        run_name = run_info["run_name"]
        
        try:
            # Check if table results already exist and have cost data
            if run_info["has_table_results"]:
                cached_stats = self.run_manager.load_table_results(run_name)
                # If cached stats don't have cost data, recalculate
                if "cost_per_image" not in cached_stats or "total_cost" not in cached_stats:
                    print(f"  Updating cached results with cost data for {run_name}")
                    # Continue to recalculate with cost data
                else:
                    return cached_stats
            
            # Load results file
            if not run_info["has_results"]:
                print(f"  Warning: No results file for run {run_name} (skipping)")
                return None
            
            results = self.run_manager.load_results(run_name)
            if not results:
                print(f"  Warning: Empty results for run {run_name} (skipping)")
                return None
            
            # Create temporary results file for processing
            temp_results_file = f"/tmp/{run_name}_results.json"
            with open(temp_results_file, 'w') as f:
                import json
                json.dump(results, f)
            
            # Process using existing string matching functions
            df_llm = get_llm_ids_and_fullnames(temp_results_file)
            df_test = get_llm_distances(df_llm, doc_info_file, test_ids_file)
            df_matching = get_matches(df_test)
            
            # Calculate statistics with failure-aware logic
            stats = self._calculate_stats(df_matching, df_test, results)
            
            # Add actual cost information based on real token usage
            cost_data = self._calculate_actual_costs(run_info, results)
            stats.update(cost_data)
            
            # Save table results for future use
            self.run_manager.save_table_results(run_name, stats)
            
            # Cleanup
            os.remove(temp_results_file)
            
            return stats
            
        except Exception as e:
            print(f"  Error processing run {run_name}: {e}")
            return None
    
    def _empty_stats(self) -> Dict:
        """Return empty statistics."""
        return {
            "digit_top1": 0.0,
            "id_top1": 0.0,
            "lastname_top1": 0.0,
            "id_avg_lev": 0.0,
            "lastname_avg_lev": 0.0,
            "docs_detected": 0.0,
            "docs_detected_count": 0,
            "cost_per_image": 0.0,
            "total_cost": 0.0,
        }
    
    def _calculate_stats(self, df_matching: pd.DataFrame, df_test: pd.DataFrame, raw_results: Dict = None) -> Dict:
        """Calculate benchmark statistics from matching results."""
        # Get best matches per document
        df_best_match = df_matching.loc[df_matching.groupby("doc")["id_distance"].idxmin()]
        
        # Calculate digit accuracy for documents with UFID data (page 3)
        df_matched_ids = pd.merge(
            df_best_match,
            df_test[df_test.page == 3],
            on=("doc", "student_full_name"),
            how="left"
        )[["student_id_x", "llm_id", "id_distance_x"]].rename(
            columns={"student_id_x": "student_id", "id_distance_x": "id_distance"}
        )
        
        # Digit-level accuracy
        if len(df_matched_ids) > 0:
            UNI_ID_LENGTH = 8
            df_matched_ids["student_id"] = df_matched_ids["student_id"].apply(str).apply(lambda x: x.zfill(UNI_ID_LENGTH))
            df_matched_ids["llm_id"] = df_matched_ids["llm_id"].apply(str).apply(lambda x: x[:UNI_ID_LENGTH].zfill(UNI_ID_LENGTH))
            
            df_ids = df_matched_ids["student_id"].apply(lambda x: pd.Series(list(x)))
            df_ids = df_ids.stack().reset_index(level=1, drop=True).to_frame("digit")
            df_ids["digit"] = df_ids["digit"].astype(int)
            
            df_llm_ids = df_matched_ids["llm_id"].apply(lambda x: pd.Series(list(x)))
            df_llm_ids = df_llm_ids.stack().reset_index(level=1, drop=True).to_frame("digit")
            df_llm_ids["digit"] = df_llm_ids["digit"].astype(int)
            
            df_ids["llm_digit"] = df_llm_ids["digit"]
            df_ids["match"] = df_ids["digit"] == df_ids["llm_digit"]
            digit_top1 = df_ids["match"].mean() * 100
        else:
            digit_top1 = 0.0
        
        # Calculate metrics
        if len(df_best_match) > 0:
            id_top1 = (df_best_match["id_distance"] == 0).sum() / len(df_best_match) * 100
            lastname_top1 = (df_best_match["lastname_distance"] == 0).sum() / len(df_best_match) * 100
            id_avg_lev = df_best_match["id_distance"].mean()
            lastname_avg_lev = df_best_match["lastname_distance"].mean()
        else:
            id_top1 = 0.0
            lastname_top1 = 0.0 
            id_avg_lev = 0.0
            lastname_avg_lev = 0.0
        
        # Calculate docs detected more accurately
        total_expected_docs = 32
        successful_docs = len(df_best_match)
        
        # If we have raw results, count parsing failures and API errors
        failed_docs = 0
        if raw_results:
            failed_images = []
            for filepath, result_list in raw_results.items():
                if result_list and len(result_list) > 0:
                    result = result_list[0]
                    if any(key.startswith('_') for key in result.keys()):
                        failed_images.append(filepath)
            # Count documents with failed page 1 or page 3 processing
            failed_doc_nums = set()
            for filepath in failed_images:
                if 'doc-' in filepath and '-page-' in filepath:
                    doc_num = filepath.split('doc-')[1].split('-page-')[0]
                    failed_doc_nums.add(int(doc_num))
            failed_docs = len(failed_doc_nums)
        
        docs_detected = successful_docs / total_expected_docs * 100
        docs_detected_count = successful_docs
        
        return {
            "digit_top1": digit_top1,
            "id_top1": id_top1,
            "lastname_top1": lastname_top1,
            "id_avg_lev": id_avg_lev,
            "lastname_avg_lev": lastname_avg_lev,
            "docs_detected": docs_detected,
            "docs_detected_count": docs_detected_count,
        }

    def _collect_matching_runs(self, run_patterns: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Collect runs that match optional patterns."""
        all_runs: List[Dict[str, Any]] = []
        if run_patterns:
            for pattern in run_patterns:
                runs = self.run_manager.list_runs(pattern)
                all_runs.extend(runs)
        else:
            all_runs = self.run_manager.list_runs()
        return all_runs

    def _is_run_eligible_for_cohort(self, run_info: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Return whether a run should participate in latest-cohort aggregation."""
        run_name = run_info.get("run_name", "unknown")
        config = run_info.get("config", {})
        runtime_value = config.get("environment", {}).get("runtime")

        if not run_info.get("has_results"):
            return False, "missing results.json"

        if isinstance(runtime_value, str) and runtime_value.strip().upper() == "TBD":
            return False, "run incomplete (runtime=TBD)"

        try:
            results = self.run_manager.load_results(run_name)
        except Exception as exc:
            return False, f"failed to load results: {type(exc).__name__}"

        if not isinstance(results, dict) or not results:
            return False, "empty/invalid results payload"

        for result_entries in results.values():
            if not result_entries or not isinstance(result_entries[0], dict):
                return False, "invalid result entry"

            entry = result_entries[0]
            for marker in self.INFERENCE_FAILURE_MARKERS:
                marker_value = entry.get(marker)
                if marker_value not in (None, False, 0, ""):
                    return False, f"inference failure marker {marker}"

        return True, None

    def _bootstrap_median_ci(
        self,
        values: List[float],
        *,
        n_resamples: int = 1000,
        confidence: float = 0.95,
        seed: int = 0,
    ) -> Optional[Tuple[float, float]]:
        """Return bootstrap CI for median when n>=3."""
        if len(values) < 3:
            return None

        rng = random.Random(seed)
        medians: List[float] = []
        n = len(values)
        for _ in range(n_resamples):
            sample = [values[rng.randrange(n)] for _ in range(n)]
            medians.append(statistics.median(sample))

        medians.sort()
        alpha = (1.0 - confidence) / 2.0
        lo_idx = max(0, int(alpha * n_resamples))
        hi_idx = min(n_resamples - 1, int((1.0 - alpha) * n_resamples) - 1)
        return medians[lo_idx], medians[hi_idx]

    def _aggregate_cohort_stats(self, cohort_entries: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Aggregate per-run stats into cohort median and optional CIs."""
        if not cohort_entries:
            return None

        metrics = [
            "digit_top1",
            "id_top1",
            "lastname_top1",
            "id_avg_lev",
            "lastname_avg_lev",
            "docs_detected",
            "docs_detected_count",
            "cost_per_image",
            "total_cost",
        ]

        aggregated: Dict[str, Any] = {
            "n_runs": len(cohort_entries),
            "ci": {},
            "run_names": [entry["run_info"]["run_name"] for entry in cohort_entries],
        }

        for metric in metrics:
            values: List[float] = []
            for entry in cohort_entries:
                value = entry["stats"].get(metric)
                if value is None:
                    continue
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    continue
            if not values:
                continue
            aggregated[metric] = float(statistics.median(values))
            ci = self._bootstrap_median_ci(values)
            if ci is not None:
                aggregated["ci"][metric] = [float(ci[0]), float(ci[1])]

        runtime_values: List[float] = []
        for entry in cohort_entries:
            config = entry["run_info"]["config"]
            runtime_seconds = config.get("additional", {}).get("actual_runtime_seconds")
            if isinstance(runtime_seconds, (int, float)):
                runtime_values.append(float(runtime_seconds))
        if runtime_values:
            aggregated["runtime_seconds"] = float(statistics.median(runtime_values))

        return aggregated

    def _format_metric_cell(
        self,
        stats: Dict[str, Any],
        metric: str,
        *,
        decimals: int = 2,
        prefix: str = "",
        suffix: str = "",
    ) -> str:
        """Format metric value with optional CI and cohort size."""
        value = stats.get(metric)
        if value is None:
            return "N/A"
        n_runs = int(stats.get("n_runs", 1) or 1)

        def _fmt(number: float) -> str:
            return f"{prefix}{number:.{decimals}f}{suffix}"

        base = _fmt(float(value))
        ci = stats.get("ci", {}).get(metric)
        if isinstance(ci, list) and len(ci) == 2 and n_runs >= 3:
            return f"{base} [{_fmt(float(ci[0]))}, {_fmt(float(ci[1]))}] (n={n_runs})"
        if n_runs > 1:
            return f"{base} (n={n_runs})"
        return base

    def _format_docs_detected_cell(self, stats: Dict[str, Any]) -> str:
        value = stats.get("docs_detected")
        count = stats.get("docs_detected_count")
        if value is None or count is None:
            return "N/A"
        n_runs = int(stats.get("n_runs", 1) or 1)
        count_int = int(round(float(count)))
        base = f"{float(value):.2f}% ({count_int}/32)"
        ci = stats.get("ci", {}).get("docs_detected")
        if isinstance(ci, list) and len(ci) == 2 and n_runs >= 3:
            return f"{base} [{float(ci[0]):.2f}%, {float(ci[1]):.2f}%] (n={n_runs})"
        if n_runs > 1:
            return f"{base} (n={n_runs})"
        return base

    def _format_runtime_cell(self, data: Dict[str, Any]) -> str:
        stats = data["stats"]
        runtime_seconds = stats.get("runtime_seconds")
        n_runs = int(stats.get("n_runs", 1) or 1)
        if isinstance(runtime_seconds, (int, float)):
            runtime_text = format_runtime(float(runtime_seconds))
            if n_runs > 1:
                return f"{runtime_text} (n={n_runs})"
            return runtime_text

        runtime_text = data["run_info"]["config"]["environment"].get("runtime", "Unknown")
        if n_runs > 1 and runtime_text != "Unknown":
            return f"{runtime_text} (n={n_runs})"
        return runtime_text

    def build_run_stats(
        self,
        run_patterns: Optional[List[str]] = None,
        doc_info_file: str = "imgs/q11/doc_info.csv",
        test_ids_file: str = "tests/data/test_ids.csv",
        *,
        cohort_window_hours: float = 24.0,
        debug_cohorts: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Build aggregated run stats grouped by latest cohort per model."""
        discovered_runs = self._collect_matching_runs(run_patterns)
        if not discovered_runs:
            return {}

        eligible_runs: List[Dict[str, Any]] = []
        excluded_runs: List[Tuple[str, str]] = []
        for run_info in discovered_runs:
            is_eligible, reason = self._is_run_eligible_for_cohort(run_info)
            if is_eligible:
                eligible_runs.append(run_info)
            else:
                excluded_runs.append((run_info.get("run_name", "unknown"), reason or "ineligible"))

        if debug_cohorts and excluded_runs:
            print("Excluded runs from cohort selection:")
            for run_name, reason in excluded_runs:
                print(f"- {run_name}: {reason}")

        if not eligible_runs:
            print("No eligible runs found after filtering incomplete/failed runs.")
            return {}

        cohorts = select_latest_cohorts(
            eligible_runs,
            model_key_getter=self._get_model_key,
            window_hours=cohort_window_hours,
        )

        if debug_cohorts:
            print(format_cohort_debug_report(cohorts))

        unknown_models = [
            model_key
            for model_key in cohorts
            if model_key not in self.model_metadata["models"]
        ]
        self._review_unknown_models(unknown_models)

        run_stats: Dict[str, Dict[str, Any]] = {}
        for model_key, cohort in cohorts.items():
            anchor_name = cohort.anchor_run["run_name"]
            if len(cohort.runs) > 1:
                print(
                    f"Processing cohort: {anchor_name} "
                    f"(model={model_key}, runs={len(cohort.runs)})"
                )
            else:
                print(f"Processing run: {anchor_name}")

            cohort_entries: List[Dict[str, Any]] = []
            for run_info in cohort.runs:
                stats = self.compute_run_stats(run_info, doc_info_file, test_ids_file)
                if not stats:
                    print(f"  Skipping {run_info['run_name']} (no usable results)")
                    continue
                cohort_entries.append(
                    {
                        "run_info": run_info,
                        "stats": stats,
                    }
                )

            aggregated = self._aggregate_cohort_stats(cohort_entries)
            if not aggregated:
                print(f"  Skipping {anchor_name} (no usable cohort stats)")
                continue

            run_stats[model_key] = {
                "run_info": cohort.anchor_run,
                "stats": aggregated,
                "cohort_runs": cohort_entries,
            }

            print(f"  digit_top1: {aggregated.get('digit_top1', 0.0):.2f}%")
            print(f"  id_top1: {aggregated.get('id_top1', 0.0):.2f}%")
            print(f"  lastname_top1: {aggregated.get('lastname_top1', 0.0):.2f}%")

        return run_stats
    
    def generate_table_for_runs(self, run_patterns: Optional[List[str]] = None, 
                               doc_info_file: str = "imgs/q11/doc_info.csv",
                               test_ids_file: str = "tests/data/test_ids.csv",
                               output_format: str = "rich",
                               cohort_window_hours: float = 24.0,
                               debug_cohorts: bool = False) -> str:
        """Generate benchmark table for specified run patterns."""
        run_stats = self.build_run_stats(
            run_patterns=run_patterns,
            doc_info_file=doc_info_file,
            test_ids_file=test_ids_file,
            cohort_window_hours=cohort_window_hours,
            debug_cohorts=debug_cohorts,
        )
        if not run_stats:
            print("No runs found matching the criteria")
            return ""
        
        # Generate table based on format
        if output_format == "markdown":
            return self._generate_markdown_table(run_stats)
        elif output_format == "rich":
            # For rich format, display table and return markdown for file output
            rich_table = self._generate_rich_table(run_stats)
            self.console.print("\n[bold blue]🏆 BENCHMARK RESULTS[/bold blue]")
            self.console.print(rich_table)
            return self._generate_markdown_table(run_stats)  # Still return markdown for file output
        else:
            raise ValueError(f"Unknown output format: {output_format}")
    
    def _generate_markdown_table(self, run_stats: Dict) -> str:
        """Generate markdown table from run statistics with models grouped by organization."""
        # Group models by organization
        models_by_org = {}
        for model_key, data in run_stats.items():
            config = data["run_info"]["config"]
            org = config['model']['org']
            model_name = config['model']['model']
            if config['model']['variant']:
                model_name += f"-{config['model']['variant']}"
            
            if org not in models_by_org:
                models_by_org[org] = []
            models_by_org[org].append((model_key, model_name, data))
        
        # Create single header row with org/model format
        headers = ["**Metric**", "**OpenCV+CNN**"]
        
        for org, models in models_by_org.items():
            for model_key, model_name, data in models:
                # Combine org and model name in a single header
                headers.append(f"**{org}**<br>{model_name}")
        
        # Create separator
        separator = [":---"] * len(headers)
        
        # Build table rows - only include configured rows
        all_rows = [headers, separator]
        
        # Create ordered list of models for consistent column ordering
        ordered_models = []
        for org, models in models_by_org.items():
            for model_key, model_name, data in models:
                ordered_models.append((model_key, data))
        
        for row_name in self.included_rows:
            row_data = [row_name, self.OPENCV_CNN_BASELINE.get(row_name, "N/A")]
            
            if row_name == "LLM model size":
                for model_key, data in ordered_models:
                    if model_key in self.model_metadata["models"]:
                        metadata = self.model_metadata["models"][model_key]
                        row_data.append(metadata.get("model_size", "??"))
                    else:
                        config = data["run_info"]["config"]["model"]
                        row_data.append(config.get("model_size", "??"))
                        
            elif row_name == "Open-weights":
                for model_key, data in ordered_models:
                    if model_key in self.model_metadata["models"]:
                        metadata = self.model_metadata["models"][model_key]
                        row_data.append("Yes" if metadata.get("open_weights", False) else "No")
                    else:
                        config = data["run_info"]["config"]["model"]
                        row_data.append("Yes" if config.get("open_weights", False) else "No")
                        
            elif row_name == "digit_top1":
                values = [self.OPENCV_CNN_BASELINE[row_name]] + [
                    self._format_metric_cell(data["stats"], "digit_top1", decimals=2, suffix="%")
                    for _, data in ordered_models
                ]
                row_data = [row_name] + self._format_best_value(values, higher_is_better=True, format_type="markdown")
                
            elif row_name == "8-digit id_top1":
                values = [self.OPENCV_CNN_BASELINE[row_name]] + [
                    self._format_metric_cell(data["stats"], "id_top1", decimals=2, suffix="%")
                    for _, data in ordered_models
                ]
                row_data = [row_name] + self._format_best_value(values, higher_is_better=True, format_type="markdown")
                
            elif row_name == "lastname_top1":
                values = [self.OPENCV_CNN_BASELINE[row_name]] + [
                    self._format_metric_cell(data["stats"], "lastname_top1", decimals=2, suffix="%")
                    for _, data in ordered_models
                ]
                row_data = [row_name] + self._format_best_value(values, higher_is_better=True, format_type="markdown")
                
            elif row_name == "ID Avg d_Lev":
                values = [self.OPENCV_CNN_BASELINE[row_name]] + [
                    self._format_metric_cell(data["stats"], "id_avg_lev", decimals=4)
                    for _, data in ordered_models
                ]
                row_data = [row_name] + self._format_best_value(values, higher_is_better=False, format_type="markdown")
                
            elif row_name == "Lastname Avg d_Lev":
                values = [self.OPENCV_CNN_BASELINE[row_name]] + [
                    self._format_metric_cell(data["stats"], "lastname_avg_lev", decimals=4)
                    for _, data in ordered_models
                ]
                row_data = [row_name] + self._format_best_value(values, higher_is_better=False, format_type="markdown")
                
            elif row_name == "Docs detected":
                values = [self.OPENCV_CNN_BASELINE[row_name]] + [
                    self._format_docs_detected_cell(data["stats"])
                    for _, data in ordered_models
                ]
                row_data = [row_name] + self._format_best_value(values, higher_is_better=True, format_type="markdown")
                
            elif row_name == "Runtime":
                values = [self.OPENCV_CNN_BASELINE[row_name]] + [
                    self._format_runtime_cell(data)
                    for _, data in ordered_models
                ]
                row_data = [row_name] + self._format_best_value(values, higher_is_better=False, format_type="markdown")
                
            elif row_name == "Cost per image":
                values = [self.OPENCV_CNN_BASELINE[row_name]] + [
                    self._format_metric_cell(data["stats"], "cost_per_image", decimals=6, prefix="$")
                    for _, data in ordered_models
                ]
                row_data = [row_name] + self._format_best_value(values, higher_is_better=False, format_type="markdown")
                
            elif row_name == "Total cost":
                values = [self.OPENCV_CNN_BASELINE[row_name]] + [
                    self._format_metric_cell(data["stats"], "total_cost", decimals=4, prefix="$")
                    for _, data in ordered_models
                ]
                row_data = [row_name] + self._format_best_value(values, higher_is_better=False, format_type="markdown")
            
            all_rows.append(row_data)
        
        # Format as markdown table
        table_lines = []
        for row in all_rows:
            if row == separator:
                line = "|" + "|".join(row) + "|"
            else:
                line = "| " + " | ".join(row) + " |"
            table_lines.append(line)
        
        return "\n".join(table_lines)
    
    def _extract_numeric_value(self, value_str: str) -> float:
        """Extract numeric value from string for comparison."""
        if not value_str or value_str in ["N/A", "??", "Unknown"]:
            return float('-inf')  # Non-comparable values get lowest priority
        
        # Extract percentage values
        if '%' in value_str:
            try:
                return float(value_str.split('%')[0])
            except ValueError:
                return float('-inf')
        
        # Handle runtime values - convert to minutes for comparison
        if 'minute' in value_str or 'second' in value_str:
            try:
                import re
                numbers = re.findall(r'(\d+\.?\d*)', value_str)
                if numbers:
                    value = float(numbers[0])
                    if 'second' in value_str:
                        return value / 60  # Convert seconds to minutes
                    else:  # minutes
                        return value
            except ValueError:
                return float('-inf')
        
        # Extract decimal values
        try:
            # Handle values like "90.62% (29/32)" - extract first number
            import re
            numbers = re.findall(r'(\d+\.?\d*)', value_str)
            if numbers:
                return float(numbers[0])
        except ValueError:
            pass
        
        return float('-inf')
    
    def _calculate_actual_costs(self, run_info: Dict, raw_results: Dict) -> Dict:
        """Calculate actual costs based on real token usage and cost data from API responses."""
        if not raw_results:
            return {"cost_per_image": 0.0, "total_cost": 0.0}
        
        # Check if any results have precise actual_cost data
        total_actual_cost = 0.0
        total_requests = 0
        has_precise_costs = False
        
        # Collect actual costs when available (from generation API)
        for filepath, result_list in raw_results.items():
            if result_list and len(result_list) > 0:
                result = result_list[0]
                token_usage = result.get("_token_usage", {})
                if token_usage and "actual_cost" in token_usage:
                    actual_cost = token_usage.get("actual_cost", 0.0)
                    if actual_cost > 0:
                        total_actual_cost += actual_cost
                        has_precise_costs = True
                        total_requests += 1
        
        if has_precise_costs and total_requests > 0:
            # Use precise costs from generation API
            cost_per_image = total_actual_cost / total_requests
            return {
                "cost_per_image": cost_per_image,
                "total_cost": total_actual_cost,
                "total_requests": total_requests
            }
        
        # Fallback: calculate from token counts and pricing rates
        config = run_info["config"]
        additional = config.get("additional", {})
        pricing = additional.get("model_pricing", {})
        
        if not pricing:
            return {"cost_per_image": 0.0, "total_cost": 0.0}
        
        # Get pricing rates (per token)
        prompt_rate = float(pricing.get("prompt", "0"))
        completion_rate = float(pricing.get("completion", "0"))
        
        # Collect token usage from API responses
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_requests = 0
        
        for filepath, result_list in raw_results.items():
            if result_list and len(result_list) > 0:
                result = result_list[0]
                token_usage = result.get("_token_usage", {})
                if token_usage:
                    total_prompt_tokens += token_usage.get("prompt_tokens", 0)
                    total_completion_tokens += token_usage.get("completion_tokens", 0)
                    total_requests += 1
        
        if total_requests == 0:
            return {"cost_per_image": 0.0, "total_cost": 0.0}
        
        # Calculate estimated costs from token counts
        total_prompt_cost = prompt_rate * total_prompt_tokens
        total_completion_cost = completion_rate * total_completion_tokens
        total_cost = total_prompt_cost + total_completion_cost
        
        # Calculate cost per image (request)
        cost_per_image = total_cost / total_requests if total_requests > 0 else 0.0
        
        return {
            "cost_per_image": cost_per_image,
            "total_cost": total_cost,
            "total_requests": total_requests,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens
        }
    
    def _format_best_value(self, values: list, higher_is_better: bool = True, format_type: str = "rich") -> list:
        """Format values with best one highlighted, handling ties properly."""
        if len(values) <= 1:  # Skip if only baseline or no data
            return values
        
        # Extract numeric values from ALL columns (including baseline at index 0)
        numeric_values = []
        for i, val in enumerate(values):
            numeric_val = self._extract_numeric_value(val)
            if numeric_val != float('-inf'):  # Only include comparable values
                numeric_values.append((i, numeric_val))
        
        # Find best value(s) - handle ties
        if not numeric_values:
            return values  # No comparable values
        
        if higher_is_better:
            best_value = max(numeric_values, key=lambda x: x[1])[1]
        else:
            best_value = min(numeric_values, key=lambda x: x[1])[1]
        
        # Find all indices with the best value (handles ties)
        best_indices = {idx for idx, val in numeric_values if val == best_value}
        
        # Format values with all tied best values highlighted
        formatted_values = []
        for i, val in enumerate(values):
            if i in best_indices:
                if format_type == "rich":
                    formatted_values.append(f"[green]{val}[/green]")
                elif format_type == "markdown":
                    formatted_values.append(f"**{val}**")
                else:
                    formatted_values.append(val)
            else:
                formatted_values.append(val)
        
        return formatted_values
    
    def _generate_rich_table(self, run_stats: Dict) -> Table:
        """Generate rich table from run statistics with models grouped by organization."""
        # Group models by organization (same as markdown)
        models_by_org = {}
        for model_key, data in run_stats.items():
            config = data["run_info"]["config"]
            org = config['model']['org']
            model_name = config['model']['model']
            if config['model']['variant']:
                model_name += f"-{config['model']['variant']}"
            
            if org not in models_by_org:
                models_by_org[org] = []
            models_by_org[org].append((model_key, model_name, data))
        
        # Create rich table
        table = Table(show_header=True, header_style="bold cyan")
        
        # Add columns - remove style from data columns to allow conditional formatting
        table.add_column("Metric", style="yellow", no_wrap=True)
        table.add_column("OpenCV+CNN", style="white", justify="right")
        
        # Add columns grouped by organization
        for org, models in models_by_org.items():
            for i, (model_key, model_name, data) in enumerate(models):
                # Format model name to wrap across lines 2-3 if needed
                if len(model_name) > 12:  # If model name is long, split it
                    # Find a good break point (prefer hyphens)
                    if '-' in model_name and len(model_name) > 15:
                        parts = model_name.split('-')
                        mid = len(parts) // 2
                        model_display = '-'.join(parts[:mid]) + '\n' + '-'.join(parts[mid:])
                    else:
                        # Break at midpoint if no good hyphen break
                        mid = len(model_name) // 2
                        model_display = model_name[:mid] + '\n' + model_name[mid:]
                else:
                    # Short model name, put on line 3 only
                    model_display = '\n' + model_name
                
                if i == 0:
                    # First model in org - show org name on line 1, model on lines 2-3
                    column_header = f"[bold]{org}[/bold]\n{model_display}"
                else:
                    # Subsequent models - show continuation symbol on line 1, model on lines 2-3  
                    column_header = f"[dim]↪[/dim]\n{model_display}"
                table.add_column(column_header, justify="right", no_wrap=False)
        
        # Create ordered list of models for consistent column ordering (same as markdown)
        ordered_models = []
        for org, models in models_by_org.items():
            for model_key, model_name, data in models:
                ordered_models.append((model_key, data))
        
        # Add non-numeric rows (no highlighting) - use metadata if available
        model_sizes = ["N/A"]
        open_weights = ["N/A"]
        
        for model_key, data in ordered_models:
            # Use metadata if available, otherwise fall back to config
            if model_key in self.model_metadata["models"]:
                metadata = self.model_metadata["models"][model_key]
                model_sizes.append(metadata.get("model_size", "??"))
                open_weights.append("Yes" if metadata.get("open_weights", False) else "No")
            else:
                # Fallback to config data
                config = data["run_info"]["config"]["model"]
                model_sizes.append(config.get("model_size", "??"))
                open_weights.append("Yes" if config.get("open_weights", False) else "No")
        
        table.add_row("LLM model size", *model_sizes)
        table.add_row("Open-weights", *open_weights)
        
        # Performance metrics with best value highlighting (higher is better)
        digit_top1_values = ["85.16%"] + [
            self._format_metric_cell(data["stats"], "digit_top1", decimals=2, suffix="%")
            for _, data in ordered_models
        ]
        table.add_row("digit_top1", *self._format_best_value(digit_top1_values, higher_is_better=True))
        
        id_top1_values = ["??"] + [
            self._format_metric_cell(data["stats"], "id_top1", decimals=2, suffix="%")
            for _, data in ordered_models
        ]
        table.add_row("8-digit id_top1", *self._format_best_value(id_top1_values, higher_is_better=True))
        
        lastname_top1_values = ["N/A"] + [
            self._format_metric_cell(data["stats"], "lastname_top1", decimals=2, suffix="%")
            for _, data in ordered_models
        ]
        table.add_row("lastname_top1", *self._format_best_value(lastname_top1_values, higher_is_better=True))
        
        # Distance metrics (lower is better)
        id_avg_lev_values = ["N/A"] + [
            self._format_metric_cell(data["stats"], "id_avg_lev", decimals=4)
            for _, data in ordered_models
        ]
        table.add_row("ID Avg d_Lev", *self._format_best_value(id_avg_lev_values, higher_is_better=False))
        
        lastname_avg_lev_values = ["N/A"] + [
            self._format_metric_cell(data["stats"], "lastname_avg_lev", decimals=4)
            for _, data in ordered_models
        ]
        table.add_row("Lastname Avg d_Lev", *self._format_best_value(lastname_avg_lev_values, higher_is_better=False))
        
        # Detection metrics (higher is better)
        docs_detected_values = ["90.62% (29/32)"] + [
            self._format_docs_detected_cell(data["stats"])
            for _, data in ordered_models
        ]
        table.add_row("Docs detected", *self._format_best_value(docs_detected_values, higher_is_better=True))
        
        # Runtime (lower is better)
        runtime_values = ["~1 second"] + [
            self._format_runtime_cell(data)
            for _, data in ordered_models
        ]
        table.add_row("Runtime", *self._format_best_value(runtime_values, higher_is_better=False))
        
        # Cost metrics (lower is better) - handle missing cost data for backward compatibility
        cost_per_image_values = ["$0.00"] + [
            self._format_metric_cell(data["stats"], "cost_per_image", decimals=6, prefix="$")
            for _, data in ordered_models
        ]
        table.add_row("Cost per image", *self._format_best_value(cost_per_image_values, higher_is_better=False))
        
        total_cost_values = ["$0.00"] + [
            self._format_metric_cell(data["stats"], "total_cost", decimals=4, prefix="$")
            for _, data in ordered_models
        ]
        table.add_row("Total cost", *self._format_best_value(total_cost_values, higher_is_better=False))
        
        return table
    
    def get_top_performers_by_category(self, run_stats: Dict, top_n: int = 2) -> Dict:
        """Get top performers by category (open-weights vs closed-source)."""
        open_weight_models = []
        closed_source_models = []
        
        for model_key, data in run_stats.items():
            if model_key in self.model_metadata["models"]:
                metadata = self.model_metadata["models"][model_key]
                if metadata.get("open_weights", False):
                    open_weight_models.append((model_key, data))
                else:
                    closed_source_models.append((model_key, data))
            else:
                # If no metadata, assume closed source (safer default)
                closed_source_models.append((model_key, data))
        
        # Sort by primary metrics: ID Levenshtein distance (lower better), then id_top1, then digit_top1
        def sort_key(x):
            stats = x[1]['stats']
            # Lower Levenshtein distance is better, so negate for sorting
            return (-stats['id_avg_lev'], stats['id_top1'], stats['digit_top1'])
        
        open_weight_models.sort(key=sort_key, reverse=True)
        closed_source_models.sort(key=sort_key, reverse=True)
        
        return {
            "open_weights": dict(open_weight_models[:top_n]),
            "closed_source": dict(closed_source_models[:top_n])
        }
    
    def generate_readme_section(self, run_stats: Dict, doc_info_file: str = "imgs/q11/doc_info.csv",
                               test_ids_file: str = "tests/data/test_ids.csv") -> str:
        """Generate README-friendly table with top performers by category."""
        
        # Get top performers
        top_performers = self.get_top_performers_by_category(run_stats, top_n=2)
        
        # Combine baseline + top performers
        readme_stats = {}
        
        # Add top open-weight models
        for model_key, data in top_performers["open_weights"].items():
            readme_stats[model_key] = data
            
        # Add top closed-source models  
        for model_key, data in top_performers["closed_source"].items():
            readme_stats[model_key] = data
        
        if not readme_stats:
            return "No models found for README table."
        
        # Generate table using existing markdown generation logic
        return self._generate_markdown_table(readme_stats)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark table from runs")
    parser.add_argument("--runs-dir", default="tests/output/runs", help="Base directory for runs")
    parser.add_argument("--patterns", nargs="*", help="Patterns to filter runs (e.g., 'glm' 'qwen')")
    parser.add_argument("--doc-info", default="imgs/q11/doc_info.csv", help="Document info CSV")
    parser.add_argument("--test-ids", default="tests/data/test_ids.csv", help="Test IDs CSV") 
    parser.add_argument("--output", help="Output markdown file")
    parser.add_argument("--format", choices=["rich", "markdown"], default="rich", 
                       help="Output format: 'rich' for terminal display (default), 'markdown' for plain markdown")
    parser.add_argument("--no-interactive", action="store_true", 
                       help="Skip interactive model review (add unknown models to needs_review list)")
    parser.add_argument("--readme", action="store_true",
                       help="Generate README-friendly table with top performers only")
    parser.add_argument("--cohort-window-hours", type=float, default=24.0,
                       help="Window (hours) for grouping latest cohorts by matching prompt hash + git commit")
    parser.add_argument("--debug-cohorts", action="store_true",
                       help="Print latest-cohort grouping details before stats generation")
    args = parser.parse_args()
    
    generator = BenchmarkTableGenerator(args.runs_dir, interactive=not args.no_interactive)
    
    if args.readme:
        # Generate README section with top performers only (from latest cohorts)
        run_stats = generator.build_run_stats(
            run_patterns=args.patterns,
            doc_info_file=args.doc_info,
            test_ids_file=args.test_ids,
            cohort_window_hours=args.cohort_window_hours,
            debug_cohorts=args.debug_cohorts,
        )
        table_markdown = generator.generate_readme_section(run_stats, args.doc_info, args.test_ids)
    else:
        # Generate full table
        table_markdown = generator.generate_table_for_runs(
            run_patterns=args.patterns,
            doc_info_file=args.doc_info,
            test_ids_file=args.test_ids,
            output_format=args.format,
            cohort_window_hours=args.cohort_window_hours,
            debug_cohorts=args.debug_cohorts,
        )
    
    if args.output:
        # Check if this is BENCHMARKS.md and needs template header
        if args.output.endswith('BENCHMARKS.md'):
            # Use relative path from this file to the template
            template_path = os.path.join(os.path.dirname(__file__), '..', 'templates', 'benchmarks.md')
            if os.path.exists(template_path):
                with open(template_path, 'r') as template_f:
                    template_content = template_f.read()
                with open(args.output, 'w') as f:
                    f.write(template_content + "\n\n" + table_markdown)
            else:
                # Fallback: template not found, just write the table
                with open(args.output, 'w') as f:
                    f.write(table_markdown)
                print(f"Warning: Template file not found at {template_path}, writing table only")
        else:
            with open(args.output, 'w') as f:
                f.write(table_markdown)
        print(f"Table written to {args.output}")
    else:
        # Only print markdown if format is markdown (rich already displayed)
        if args.format == "markdown":
            print("\n" + "="*80)
            print("BENCHMARK TABLE")
            print("="*80)
            print(table_markdown)


if __name__ == "__main__":
    main()
