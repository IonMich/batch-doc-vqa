#!/usr/bin/env python3
"""
OpenRouter inference engine and orchestration.
"""
import time
import re
import json
import requests  # type: ignore[import]
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, cast

from rich.console import Console
from rich.prompt import Confirm

from ..core import (
    RunManager,
    RunConfig,
    format_runtime,
    create_inference_progress,
    add_inference_task,
    get_imagepaths,
    get_imagepaths_from_doc_info,
    build_git_dirty_warning_lines,
)
from .api import (
    MODEL_CONFIG_OVERRIDES, 
    create_completion, 
    parse_response_content, 
    fetch_openrouter_models,
    model_supports_image_input,
    batch_update_generation_costs,
)
from .ui import interactive_config_prompt
from .defaults import DEFAULT_EXTRACTION_PRESET_ID, DEFAULT_IMAGES_DIR
from .spec import load_extraction_spec
from .extraction_adapter import build_extraction_adapter

console = Console()


def assess_repetition(text: str, *, min_tokens: int = 80) -> tuple[bool, float]:
    """Heuristically determine whether a response is highly repetitive."""
    if not text:
        return False, 0.0

    tokens = re.findall(r"\w+|[^\s\w]", text.lower())
    total_tokens = len(tokens)

    if total_tokens < min_tokens:
        return False, 0.0

    token_counts = Counter(tokens)
    unique_tokens = len(token_counts)
    if unique_tokens == 0:
        return False, 0.0

    most_common_count = token_counts.most_common(1)[0][1]
    repetition_ratio = most_common_count / total_tokens
    diversity_ratio = unique_tokens / total_tokens

    consecutive_repeats = 0.0
    if total_tokens > 1:
        consecutive_repeats = sum(1 for i in range(total_tokens - 1) if tokens[i] == tokens[i + 1]) / (total_tokens - 1)

    repetition_score = max(repetition_ratio, 1 - diversity_ratio, consecutive_repeats)
    is_repetitive = repetition_score >= 0.22 or (diversity_ratio <= 0.35 and repetition_ratio >= 0.18)

    return is_repetitive, repetition_score


def run_openrouter_inference(model_name: str, 
                            preset_id: str = DEFAULT_EXTRACTION_PRESET_ID,
                            temperature: float = 0.0,
                            max_tokens: int = 4096,
                            top_p: float = 1.0,
                            repetition_penalty: Optional[float] = None,
                            provider_order: Optional[list[str]] = None,
                            provider_allow_fallbacks: Optional[bool] = None,
                            provider_sort: Optional[str] = None,
                            provider_data_collection: Optional[str] = None,
                            provider_zdr: Optional[bool] = None,
                            model_size: Optional[str] = None,
                            open_weights: Optional[bool] = None,
                            license_info: Optional[str] = None,
                            interactive: bool = False,
                            confirm_reproducibility_warnings: bool = False,
                            skip_reproducibility_checks: bool = False,
                            concurrency: int = 1,
                            rate_limit: Optional[float] = None,
                            retry_max: int = 3,
                            retry_base_delay: float = 2.0,
                            images_dir: Optional[str] = None,
                            dataset_manifest_file: Optional[str] = None,
                            doc_info_file: Optional[str] = None,
                            pages: Optional[list[int]] = None,
                            prompt_file: Optional[str] = None,
                            schema_file: Optional[str] = None,
                            output_json: Optional[str] = None,
                            strict_schema: Optional[bool] = None):
    """Run inference using any OpenRouter vision model."""
    
    # Start timing
    start_time = time.time()

    try:
        extraction_spec = load_extraction_spec(
            preset_id=preset_id,
            prompt_file=prompt_file,
            schema_file=schema_file,
        )
    except Exception as exc:
        console.print(f"[red]❌ Failed to load extraction spec: {exc}[/red]")
        return ""

    effective_strict_schema = (
        extraction_spec.strict_schema_default
        if strict_schema is None
        else bool(strict_schema)
    )

    schema_validator: Optional[Any] = None
    if extraction_spec.mode == "custom":
        try:
            from jsonschema import Draft202012Validator
            Draft202012Validator.check_schema(extraction_spec.schema)
            schema_validator = Draft202012Validator(extraction_spec.schema)
        except Exception as exc:
            console.print(f"[red]❌ Invalid custom schema: {exc}[/red]")
            return ""

    extraction_adapter = build_extraction_adapter(
        spec=extraction_spec,
        schema_validator=schema_validator,
    )

    effective_images_dir = (
        images_dir.strip()
        if isinstance(images_dir, str) and images_dir.strip()
        else DEFAULT_IMAGES_DIR
    )
    if dataset_manifest_file and doc_info_file and dataset_manifest_file != doc_info_file:
        console.print(
            "[yellow]⚠️ Both dataset_manifest_file and doc_info_file were provided; using dataset_manifest_file.[/yellow]"
        )
    effective_dataset_manifest = dataset_manifest_file or doc_info_file
    dataset_manifest_autodiscovered = False
    if not effective_dataset_manifest:
        candidate_manifest = Path(effective_images_dir) / "doc_info.csv"
        if candidate_manifest.exists() and candidate_manifest.is_file():
            effective_dataset_manifest = str(candidate_manifest.resolve(strict=False))
            dataset_manifest_autodiscovered = True
            console.print(
                f"[dim]Auto-detected dataset manifest: {effective_dataset_manifest}[/dim]"
            )

    def _normalize_model_id(value: str) -> str:
        return value.strip().lower()

    def _model_id_matches(candidate: str, target: str) -> bool:
        c = _normalize_model_id(candidate)
        t = _normalize_model_id(target)
        if c == t:
            return True
        # Handle variants like ":free" when one side omits the suffix.
        return c.split(":", 1)[0] == t.split(":", 1)[0]

    def _fetch_model_provider_names(model_id: str) -> Optional[set[str]]:
        if "/" not in model_id:
            return None
        author, slug = model_id.split("/", 1)
        url = f"https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None

        endpoints = payload.get("data", {}).get("endpoints", [])
        if not isinstance(endpoints, list):
            return None

        providers: set[str] = set()
        for endpoint in endpoints:
            if not isinstance(endpoint, dict):
                continue
            provider_name = endpoint.get("provider_name")
            if isinstance(provider_name, str) and provider_name.strip():
                providers.add(provider_name.strip().lower())
        return providers if providers else set()

    def _fetch_model_zdr_provider_names(model_id: str) -> Optional[set[str]]:
        url = "https://openrouter.ai/api/v1/endpoints/zdr"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None

        rows = payload.get("data")
        if not isinstance(rows, list):
            return None

        providers: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            endpoint_model = row.get("model_id")
            provider_name = row.get("provider_name")
            if not isinstance(endpoint_model, str) or not isinstance(provider_name, str):
                continue
            if _model_id_matches(endpoint_model, model_id):
                providers.add(provider_name.strip().lower())
        return providers if providers else set()

    def parse_with_reasoning_fallback(
        primary_content: Optional[str],
        reasoning_content: Optional[str],
        *,
        log_image: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        parsed = None
        if isinstance(response_format, str):
            parsed = parse_response_content(primary_content or "", response_format)

        if parsed is not None:
            return parsed

        if isinstance(reasoning_content, str) and reasoning_content.strip():
            fallback = parse_response_content(reasoning_content.strip(), response_format)
            if fallback is not None and log_image:
                console.print(f"[yellow]⚠️ Using reasoning text as fallback for {log_image}[/yellow]")
            return fallback

        return None

    def build_reasoning_retry_steps(
        *,
        reasoning_capable: bool,
        include_reasoning_capable: bool,
        target_max_tokens: int,
        base_repetition_penalty: Optional[float],
    ) -> list[dict[str, Any]]:
        steps: list[dict[str, Any]] = []

        if reasoning_capable:
            for effort in ("medium", "low"):
                steps.append({
                    "label": f"reasoning_{effort}",
                    "config_updates": {
                        "max_tokens": target_max_tokens,
                        "reasoning": {"effort": effort},
                        "include_reasoning": True,
                    },
                })

            penalty_value = 1.1
            if base_repetition_penalty and base_repetition_penalty > penalty_value:
                penalty_value = base_repetition_penalty

            steps.append({
                "label": "reasoning_high_penalty",
                "config_updates": {
                    "max_tokens": target_max_tokens,
                    "reasoning": {"effort": "high"},
                    "include_reasoning": True,
                    "repetition_penalty": penalty_value,
                },
            })

        if include_reasoning_capable:
            steps.append({
                "label": "reasoning_disabled",
                "config_updates": {
                    "max_tokens": target_max_tokens,
                    "include_reasoning": False,
                },
                "remove_keys": ["reasoning"],
            })

        return steps

    # Fetch model data to get supported parameters
    models = fetch_openrouter_models()
    model_data = None
    if models:
        model_data = next((m for m in models if m.get("id") == model_name), None)
    if model_data and not model_supports_image_input(model_data):
        console.print(
            f"[red]❌ Model {model_name} is not image-capable according to OpenRouter architecture metadata.[/red]"
        )
        console.print(
            "[red]This pipeline requires image-input models; aborting before requests are sent.[/red]"
        )
        return ""
    
    # Parse model name
    if "/" in model_name:
        org, model = model_name.split("/", 1)
    else:
        org, model = "unknown", model_name
    
    # Get model-specific overrides if they exist
    overrides = cast(Dict[str, Any], MODEL_CONFIG_OVERRIDES.get(model_name, {}))
    default_schema_retry_max = 2
    schema_retry_max = overrides.get("schema_retry_max", default_schema_retry_max)
    if not isinstance(schema_retry_max, int) or schema_retry_max < 0:
        schema_retry_max = default_schema_retry_max

    effective_repetition_penalty: Optional[float] = repetition_penalty
    if effective_repetition_penalty is None:
        override_penalty = overrides.get("repetition_penalty")
        if isinstance(override_penalty, (int, float)):
            effective_repetition_penalty = float(override_penalty)
    
    # Set defaults with potential overrides
    response_format = overrides.get("response_format", "json")
    
    # Use provided max_tokens (now has default of 4096)
    if "max_tokens" in overrides:
        override_max_tokens = overrides.get("max_tokens")
        if isinstance(override_max_tokens, int):
            max_tokens = override_max_tokens

    allowed_provider_sorts = {"price", "throughput", "latency"}
    allowed_data_collection = {"allow", "deny"}
    data_collection_explicit = provider_data_collection is not None

    if provider_data_collection is None:
        provider_data_collection = "deny"
    if provider_zdr is None:
        provider_zdr = True

    normalized_provider_order: list[str] = []
    if provider_order:
        seen_providers: set[str] = set()
        for provider_slug in provider_order:
            normalized = str(provider_slug).strip().lower()
            if not normalized or normalized in seen_providers:
                continue
            seen_providers.add(normalized)
            normalized_provider_order.append(normalized)

    normalized_provider_sort: Optional[str] = None
    if isinstance(provider_sort, str):
        candidate_sort = provider_sort.strip().lower()
        if candidate_sort in allowed_provider_sorts:
            normalized_provider_sort = candidate_sort

    normalized_data_collection: Optional[str] = None
    if isinstance(provider_data_collection, str):
        candidate = provider_data_collection.strip().lower()
        if candidate in allowed_data_collection:
            normalized_data_collection = candidate

    normalized_provider_zdr: Optional[bool] = None
    if provider_zdr is not None:
        normalized_provider_zdr = bool(provider_zdr)

    if normalized_provider_zdr is True:
        all_providers = _fetch_model_provider_names(model_name)
        zdr_providers = _fetch_model_zdr_provider_names(model_name)

        if all_providers is not None and zdr_providers is not None and len(all_providers) > 0:
            safe_providers = sorted(all_providers.intersection(zdr_providers))
            unsafe_providers = sorted(all_providers.difference(zdr_providers))

            if safe_providers:
                console.print(
                    f"[green]🔐 Privacy-safe providers (ZDR) for this model: {', '.join(safe_providers)}[/green]"
                )
                if unsafe_providers:
                    console.print(
                        f"[dim]Excluded non-ZDR providers: {', '.join(unsafe_providers)}[/dim]"
                    )
            else:
                console.print(
                    "[yellow]⚠️ No ZDR endpoints are currently available for this model.[/yellow]"
                )
                if interactive:
                    proceed_relaxed = Confirm.ask(
                        "Continue by relaxing privacy routing (provider.data_collection=allow, provider.zdr=false)?",
                        default=False,
                    )
                    if not proceed_relaxed:
                        console.print("[yellow]Run cancelled before start.[/yellow]")
                        return ""

                    normalized_provider_zdr = False
                    if not data_collection_explicit:
                        normalized_data_collection = "allow"
                    console.print(
                        "[yellow]Proceeding with relaxed privacy routing by explicit user confirmation.[/yellow]"
                    )
                else:
                    console.print(
                        "[red]❌ Privacy routing check failed: no ZDR endpoints are available for this model.[/red]"
                    )
                    console.print(
                        "[red]Non-interactive mode will not relax privacy defaults automatically.[/red]"
                    )
                    console.print(
                        "[yellow]Override explicitly with --no-provider-zdr and/or --provider-data-collection allow if desired.[/yellow]"
                    )
                    return ""

    provider_routing_effective: Dict[str, Any] = {}
    if normalized_provider_order:
        provider_routing_effective["order"] = normalized_provider_order
    if provider_allow_fallbacks is not None:
        provider_routing_effective["allow_fallbacks"] = provider_allow_fallbacks
    if normalized_provider_sort is not None:
        provider_routing_effective["sort"] = normalized_provider_sort
    if normalized_data_collection is not None:
        provider_routing_effective["data_collection"] = normalized_data_collection
    if normalized_provider_zdr is not None:
        provider_routing_effective["zdr"] = normalized_provider_zdr

    provider_routing_requested = {
        "order": normalized_provider_order,
        "allow_fallbacks": provider_allow_fallbacks,
        "sort": normalized_provider_sort,
        "data_collection": normalized_data_collection,
        "zdr": normalized_provider_zdr,
    }
    
    # Extract supported parameters from model data
    supported_parameters = []
    model_context_length = "Unknown"
    model_pricing = {}
    
    if model_data:
        supported_parameters = model_data.get("supported_parameters", [])
        model_context_length = model_data.get("context_length", "Unknown")
        model_pricing = model_data.get("pricing", {})
    
    # Interactive configuration if missing key info and interactive mode
    if interactive and (model_size is None or open_weights is None or license_info is None):
        interactive_config = interactive_config_prompt(model_name)
        model_size = model_size or interactive_config["model_size"]
        open_weights = open_weights if open_weights is not None else interactive_config["open_weights"]
        license_info = license_info or interactive_config["license_info"]
    
    # Set final defaults
    if model_size is None:
        override_model_size = overrides.get("model_size")
        model_size = override_model_size if isinstance(override_model_size, str) else "Unknown"
    open_weights = open_weights if open_weights is not None else False
    license_info = license_info or "Varies by provider"
    
    selected_pages = list(pages) if pages else list(extraction_spec.default_pages)

    # Create run configuration (runtime will be updated after inference)
    config = RunConfig(
        org=org,
        model=model,
        model_size=model_size,
        open_weights=open_weights,
        license_info=license_info,
        api_provider="OpenRouter (Router)",
        use_structured_output=True,
        use_regex_patterns=False,
        temperature=temperature,
        max_tokens=max_tokens,
        runtime_environment="TBD",  # Will be updated with actual runtime
        additional_config={
            "base_url": "https://openrouter.ai/api/v1/chat/completions",
            "endpoint_type": "chat_completions",
            "response_format": response_format,
            "model_name": model_name,
            "concurrency": int(max(1, concurrency)),
            "rate_limit": rate_limit,
            "retry_max": int(max(0, retry_max)),
            "retry_base_delay": float(max(0.0, retry_base_delay)),
            "schema_retry_max": int(schema_retry_max),
            "cost_fetch_max_workers": int(max(1, concurrency)),
            "top_p": top_p,
            "repetition_penalty": effective_repetition_penalty,
            "provider_routing_requested": provider_routing_requested,
            "provider_routing_effective": provider_routing_effective,
            "images_dir": effective_images_dir,
            "dataset_manifest_file": effective_dataset_manifest,
            "doc_info_file": effective_dataset_manifest,
            "dataset_manifest_autodiscovered": dataset_manifest_autodiscovered,
            "pages": selected_pages,
            "prompt_template": extraction_spec.prompt_text,
            "prompt_source_file": extraction_spec.prompt_source,
            "preset_id": extraction_spec.preset_id,
            "extraction_mode": extraction_spec.mode,
            "schema_source_file": extraction_spec.schema_source,
            "schema_hash": extraction_spec.schema_hash,
            "strict_schema": effective_strict_schema,
            "output_json": output_json,
            "actual_model_providers": set(),  # Will track actual model providers used
        }
    )

    for warning_line in build_git_dirty_warning_lines(config):
        console.print(warning_line)

    if config.git_dirty_relevant and not skip_reproducibility_checks:
        if not confirm_reproducibility_warnings:
            console.print(
                "[red]❌ Reproducibility check failed: relevant uncommitted changes detected.[/red]"
            )
            console.print(
                "[red]Non-interactive runs require a clean reproducibility state.[/red]"
            )
            console.print(
                "[yellow]Use --skip-reproducibility-checks to override this guard explicitly.[/yellow]"
            )
            return ""

        console.print("\n[bold yellow]Pre-run reproducibility check[/bold yellow]")
        console.print(
            "[yellow]This run has reproducibility-relevant uncommitted changes "
            "and may not be comparable to clean runs.[/yellow]"
        )
        proceed_anyway = Confirm.ask(
            "Start this run anyway?",
            default=False,
        )
        if not proceed_anyway:
            console.print("[yellow]Run cancelled before start.[/yellow]")
            return ""
    
    # Create run directory
    manager = RunManager()
    run_dir = manager.create_run_directory(config)
    reasoning_log_path = Path(run_dir) / "failed_reasoning.log"

    def log_reasoning_trace(imagepath: str,
                             message: Dict[str, Any],
                             response_json: Dict[str, Any],
                             *,
                             retry_stage: str) -> None:
        reasoning_text = message.get("reasoning")
        if not reasoning_text:
            return

        log_entry = {
            "timestamp": time.time(),
            "image": imagepath,
            "retry_stage": retry_stage,
            "finish_reason": message.get("finish_reason") or response_json.get("choices", [{}])[0].get("finish_reason"),
            "native_finish_reason": response_json.get("choices", [{}])[0].get("native_finish_reason"),
            "usage": response_json.get("usage", {}),
            "reasoning": reasoning_text,
            "content_preview": (message.get("content") or "")[:200].strip(),
        }

        try:
            with open(reasoning_log_path, "a", encoding="utf-8") as log_file:
                log_file.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            console.print(f"[dim]Saved reasoning trace for {imagepath} to {reasoning_log_path.name}[/dim]")
        except Exception as exc:
            console.print(f"[yellow]⚠️ Failed to write reasoning log: {exc}[/yellow]")
    
    print(f"Starting OpenRouter inference run: {config.run_name}")
    print(f"Model: {model_name}")
    print(f"Run directory: {run_dir}")
    if provider_routing_effective:
        print(f"Provider routing: {provider_routing_effective}")
    else:
        print("Provider routing: OpenRouter default")
    
    # Setup inference parameters
    if effective_dataset_manifest:
        imagepaths = get_imagepaths_from_doc_info(
            effective_dataset_manifest,
            images_dir=effective_images_dir,
            pages=selected_pages,
        )
    else:
        pattern = r"doc-\d+-page-[" + "".join([str(p) for p in selected_pages]) + "]-[A-Z0-9]+.png"
        imagepaths = get_imagepaths(effective_images_dir, pattern)
    
    inference_config: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    if provider_routing_effective:
        inference_config["provider"] = provider_routing_effective

    max_no_choice_retries = overrides.get("no_choice_retries", 2)
    if not isinstance(max_no_choice_retries, int) or max_no_choice_retries < 0:
        max_no_choice_retries = 2

    include_reasoning_override = overrides.get("include_reasoning")
    reasoning_override = overrides.get("reasoning")

    if include_reasoning_override is not None:
        inference_config["include_reasoning"] = include_reasoning_override

    if reasoning_override is not None:
        inference_config["reasoning"] = reasoning_override

    reasoning_capable = "reasoning" in supported_parameters or reasoning_override is not None
    include_reasoning_capable = reasoning_capable or "include_reasoning" in supported_parameters or include_reasoning_override is not None

    if effective_repetition_penalty is not None:
        inference_config["repetition_penalty"] = effective_repetition_penalty
    
    # Parallel processing setup
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    class RateLimiter:
        def __init__(self, rate: Optional[float]):
            self.rate = rate
            self.lock = threading.Lock()
            self.next_time = time.monotonic()

        def acquire(self):
            if not self.rate or self.rate <= 0:
                return
            with self.lock:
                now = time.monotonic()
                wait = max(0.0, self.next_time - now)
                base = max(self.next_time, now)
                interval = 1.0 / self.rate
                self.next_time = base + interval
            if wait > 0:
                time.sleep(wait)

    rate_limiter = RateLimiter(rate_limit)

    def summarize_failure_reason(result_entries: list[Dict[str, Any]]) -> str:
        if not result_entries:
            return "unknown failure"

        entry = result_entries[0]
        if not isinstance(entry, dict):
            return "unknown failure"

        if entry.get("_schema_failed"):
            errors = entry.get("_schema_errors")
            if isinstance(errors, list) and errors:
                return f"schema: {errors[0]}"
            return "schema validation failed"
        if entry.get("_parse_failed"):
            return "parse failed"
        if entry.get("_no_response"):
            return "no response from model"
        if entry.get("_empty_response"):
            return "empty response"
        if entry.get("_server_error"):
            code = entry.get("_api_error")
            return f"server error {code}" if code is not None else "server error"
        if entry.get("_api_error") is not None:
            return f"api error {entry.get('_api_error')}"
        if entry.get("_exception"):
            return f"exception: {entry.get('_exception')}"
        return "inference failed"

    base_result_entry = extraction_adapter.base_result_entry

    def is_retryable_transport_exception(exc: Exception) -> bool:
        if isinstance(exc, (TimeoutError, ConnectionError, OSError)):
            return True

        error_text = str(exc).lower()
        retryable_markers = (
            "connection reset",
            "connection aborted",
            "connection broken",
            "read timed out",
            "timed out",
            "timeout",
            "temporarily unavailable",
            "remote end closed",
            "failed to establish a new connection",
            "connection refused",
            "name resolution",
            "name or service not known",
            "nodename nor servname",
        )
        return any(marker in error_text for marker in retryable_markers)

    def guarded_create_completion(
        local_model_name: str,
        local_config: Dict[str, Any],
        img_path: str,
        *,
        prompt_text: Optional[str] = None,
    ):
        max_transport_retries = max(0, retry_max)
        base_delay = max(0.1, retry_base_delay)
        attempt = 0

        while True:
            rate_limiter.acquire()
            try:
                if prompt_text is None:
                    prompt_text = extraction_spec.prompt_text
                return create_completion(local_model_name, local_config, img_path, prompt_text=prompt_text)
            except Exception as exc:
                retryable = is_retryable_transport_exception(exc)
                if not retryable or attempt >= max_transport_retries:
                    if retryable and attempt > 0:
                        raise RuntimeError(
                            f"Transport error after {attempt + 1} attempts: {exc}"
                        ) from exc
                    raise

                delay = min(60.0, base_delay * (2 ** attempt))
                time.sleep(delay)
                attempt += 1

    # Run inference with rich progress tracking
    results = defaultdict(list)
    total_images = len(imagepaths)
    if total_images == 0:
        console.print("[red]❌ No images matched the selected dataset/pages.[/red]")
        return ""
    completed_images = 0
    successful_images = 0
    repetition_event_scores: Dict[str, float] = {}
    failed_images: list[tuple[str, str]] = []
    provider_lock = threading.Lock()

    # Per-image worker encapsulating the existing logic. Avoids disk writes and progress updates.
    def process_image(imagepath: str):
        local_results: list[Dict[str, Any]] = []
        local_providers: set[str] = set()
        local_rep_score: float = 0.0
        status_msg: str = ""
        image_started_epoch = time.time()
        image_started_at_utc = datetime.now(timezone.utc).isoformat()

        def finalize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
            """Attach per-image wall-clock timing to payload and result entries."""
            image_finished_at_utc = datetime.now(timezone.utc).isoformat()
            elapsed_seconds = max(0.0, time.time() - image_started_epoch)
            payload["_image_timing"] = {
                "started_at_utc": image_started_at_utc,
                "finished_at_utc": image_finished_at_utc,
                "elapsed_seconds": elapsed_seconds,
            }

            result_entries = payload.get("results")
            if isinstance(result_entries, list):
                for entry in result_entries:
                    if not isinstance(entry, dict):
                        continue
                    entry["_timing"] = {
                        "started_at_utc": image_started_at_utc,
                        "finished_at_utc": image_finished_at_utc,
                        "elapsed_seconds": elapsed_seconds,
                    }
            return payload

        # Copy base inference config for this image
        local_inference_config: Dict[str, Any] = dict(inference_config)
        try:
            response = guarded_create_completion(model_name, local_inference_config, imagepath)

            # Handle rate limiting with a single retry
            if response.status_code == 429:
                time.sleep(10)
                response = guarded_create_completion(model_name, local_inference_config, imagepath)

            # Hard-stop critical errors
            if response.status_code == 402:
                return finalize_payload({
                    "critical_error": (402, "Insufficient funds"),
                    "imagepath": imagepath,
                })
            if response.status_code == 401:
                return finalize_payload({
                    "critical_error": (401, "Invalid API key"),
                    "imagepath": imagepath,
                })

            # Retry transient server errors (5xx) per image with backoff
            server_attempts = 0
            while response.status_code >= 500 and server_attempts < max(0, retry_max):
                delay = min(60.0, retry_base_delay * (2 ** server_attempts))
                time.sleep(delay)
                response = guarded_create_completion(model_name, local_inference_config, imagepath)
                server_attempts += 1

            # If still a server error after retries, record non-critical failure and continue
            if response.status_code >= 500:
                try:
                    error_usage = response.json().get("usage", {})
                except Exception:
                    error_usage = {}
                server_error_entry = base_result_entry()
                server_error_entry.update({
                    "_api_error": response.status_code,
                    "_server_error": True,
                    "_token_usage": {
                        "prompt_tokens": error_usage.get("prompt_tokens", 0),
                        "completion_tokens": error_usage.get("completion_tokens", 0),
                        "total_tokens": error_usage.get("total_tokens", 0)
                    }
                })
                local_results.append(server_error_entry)
                status_msg = f"[red]Server error {response.status_code} - continuing[/red]"
                return finalize_payload({
                    "imagepath": imagepath,
                    "results": local_results,
                    "providers": list(local_providers),
                    "rep_score": local_rep_score,
                    "status_msg": status_msg,
                    "success": False,
                })

            if response.status_code >= 400:
                try:
                    error_usage = response.json().get("usage", {})
                except Exception:
                    error_usage = {}
                api_error_entry = base_result_entry()
                api_error_entry.update({
                    "_api_error": response.status_code,
                    "_token_usage": {
                        "prompt_tokens": error_usage.get("prompt_tokens", 0),
                        "completion_tokens": error_usage.get("completion_tokens", 0),
                        "total_tokens": error_usage.get("total_tokens", 0)
                    }
                })
                local_results.append(api_error_entry)
                status_msg = f"[red]API Error {response.status_code}[/red]"
                return finalize_payload({
                    "imagepath": imagepath,
                    "results": local_results,
                    "providers": list(local_providers),
                    "rep_score": local_rep_score,
                    "status_msg": status_msg,
                    "success": False,
                })

            response_json = response.json()
            if "provider" in response_json:
                local_providers.add(response_json["provider"])

            if "choices" not in response_json or not response_json["choices"]:
                missing_choices = True
                no_choice_attempts = 0

                while missing_choices and no_choice_attempts < max_no_choice_retries:
                    no_choice_attempts += 1
                    time.sleep(min(5 * no_choice_attempts, 15))
                    retry_response = guarded_create_completion(model_name, local_inference_config, imagepath)
                    if retry_response.status_code >= 500:
                        continue
                    if retry_response.status_code == 429:
                        time.sleep(10)
                        retry_response = guarded_create_completion(model_name, local_inference_config, imagepath)
                    if retry_response.status_code != 200:
                        continue
                    retry_response_json = retry_response.json()
                    if "provider" in retry_response_json:
                        local_providers.add(retry_response_json["provider"])
                    if "choices" in retry_response_json and retry_response_json["choices"]:
                        response_json = retry_response_json
                        missing_choices = False
                        break

                if missing_choices:
                    usage = response_json.get("usage", {})
                    error_details = response_json.get("error", {})
                    no_response_entry = base_result_entry()
                    no_response_entry.update({
                        "_no_response": True,
                        "_no_response_error": error_details,
                        "_token_usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                    })
                    local_results.append(no_response_entry)
                    status_msg = "[red]No response[/red]"
                    return finalize_payload({
                        "imagepath": imagepath,
                        "results": local_results,
                        "providers": list(local_providers),
                        "rep_score": local_rep_score,
                        "status_msg": status_msg,
                        "success": False,
                    })

            message = response_json["choices"][0]["message"]
            content = message.get("content", "")
            choice = response_json["choices"][0]
            reasoning_text = message.get("reasoning")

            repetition_detected = False
            repetition_score = 0.0

            def update_repetition_metrics(content_text: Optional[str], reasoning_text_val: Optional[str]) -> None:
                nonlocal repetition_detected, repetition_score
                combined = " ".join(
                    part.strip() for part in (content_text, reasoning_text_val) if isinstance(part, str) and part.strip()
                )
                if not combined:
                    return
                detected, score = assess_repetition(combined)
                if detected:
                    repetition_detected = True
                    repetition_score = max(repetition_score, score)
            update_repetition_metrics(content, reasoning_text)

            length_retry_attempts = 0
            max_length_retry_attempts = 3
            current_max_tokens = local_inference_config.get("max_tokens", max_tokens)
            override_ceiling = overrides.get("max_tokens_ceiling")
            if isinstance(override_ceiling, int) and override_ceiling > 0:
                max_tokens_ceiling = max(override_ceiling, current_max_tokens)
            else:
                max_tokens_ceiling = max(16384, current_max_tokens)

            while choice.get("finish_reason") == "length" and length_retry_attempts < max_length_retry_attempts:
                if repetition_detected:
                    break
                new_max_tokens = min(max_tokens_ceiling, current_max_tokens * 2)
                if new_max_tokens <= current_max_tokens:
                    break
                time.sleep(2)
                retry_config = {**local_inference_config, "max_tokens": new_max_tokens}
                retry_response = guarded_create_completion(model_name, retry_config, imagepath)
                if retry_response.status_code != 200:
                    break
                retry_response_json = retry_response.json()
                if "provider" in retry_response_json:
                    local_providers.add(retry_response_json["provider"])
                if "choices" not in retry_response_json or not retry_response_json["choices"]:
                    break
                retry_message = retry_response_json["choices"][0]["message"]
                content = retry_message.get("content", "")
                reasoning_text = retry_message.get("reasoning")
                choice = retry_response_json["choices"][0]
                update_repetition_metrics(content, reasoning_text)
                response_json = retry_response_json
                current_max_tokens = new_max_tokens
                local_inference_config["max_tokens"] = new_max_tokens
                length_retry_attempts += 1

            if not content and choice.get("finish_reason") is None:
                time.sleep(10)
                retry_response = guarded_create_completion(model_name, local_inference_config, imagepath)
                if retry_response.status_code == 200:
                    retry_response_json = retry_response.json()
                    if "provider" in retry_response_json:
                        local_providers.add(retry_response_json["provider"])
                    if "choices" in retry_response_json and retry_response_json["choices"]:
                        retry_message = retry_response_json["choices"][0]["message"]
                        retry_content = retry_message.get("content", "")
                        if retry_content:
                            content = retry_content
                            response_json = retry_response_json
                            choice = retry_response_json["choices"][0]
                            reasoning_text = retry_message.get("reasoning")
                            update_repetition_metrics(content, reasoning_text)
                        else:
                            empty_retry_entry = base_result_entry()
                            empty_retry_entry.update({
                                "_empty_response": True,
                                "_retry_failed": True
                            })
                            local_results.append(empty_retry_entry)
                            status_msg = "[red]Empty after retry[/red]"
                            return finalize_payload({
                                "imagepath": imagepath,
                                "results": local_results,
                                "providers": list(local_providers),
                                "rep_score": local_rep_score,
                                "status_msg": status_msg,
                                "success": False,
                            })
                else:
                    retry_failed_entry = base_result_entry()
                    retry_failed_entry.update({
                        "_empty_response": True,
                        "_retry_failed": True
                    })
                    local_results.append(retry_failed_entry)
                    status_msg = "[red]Retry failed[/red]"
                    return finalize_payload({
                        "imagepath": imagepath,
                        "results": local_results,
                        "providers": list(local_providers),
                        "rep_score": local_rep_score,
                        "status_msg": status_msg,
                        "success": False,
                    })

            json_obj = parse_with_reasoning_fallback(content, reasoning_text, log_image=imagepath)
            target_parse_tokens = min(max_tokens_ceiling, 16384)

            while json_obj is None and current_max_tokens < target_parse_tokens:
                new_max_tokens = min(target_parse_tokens, max(current_max_tokens * 2, current_max_tokens + 512))
                if new_max_tokens <= current_max_tokens:
                    break
                time.sleep(2)
                retry_config = {**local_inference_config, "max_tokens": new_max_tokens}
                retry_response = guarded_create_completion(model_name, retry_config, imagepath)
                if retry_response.status_code != 200:
                    break
                retry_response_json = retry_response.json()
                if "provider" in retry_response_json:
                    local_providers.add(retry_response_json["provider"])
                if "choices" not in retry_response_json or not retry_response_json["choices"]:
                    break
                retry_message = retry_response_json["choices"][0]["message"]
                content = retry_message.get("content", "")
                reasoning_text = retry_message.get("reasoning")
                response_json = retry_response_json
                update_repetition_metrics(content, reasoning_text)
                current_max_tokens = new_max_tokens
                local_inference_config["max_tokens"] = new_max_tokens
                json_obj = parse_with_reasoning_fallback(content, reasoning_text, log_image=imagepath)

            if json_obj is None:
                adaptive_target_tokens = max(current_max_tokens, target_parse_tokens)
                if local_inference_config.get("max_tokens", current_max_tokens) < adaptive_target_tokens:
                    local_inference_config["max_tokens"] = adaptive_target_tokens
                    current_max_tokens = adaptive_target_tokens

                adaptive_steps = build_reasoning_retry_steps(
                    reasoning_capable=reasoning_capable,
                    include_reasoning_capable=include_reasoning_capable,
                    target_max_tokens=adaptive_target_tokens,
                    base_repetition_penalty=local_inference_config.get("repetition_penalty"),
                )

                for step in adaptive_steps:
                    retry_config = dict(local_inference_config)
                    for key in step.get("remove_keys", []):
                        retry_config.pop(key, None)
                    retry_config.update(step.get("config_updates", {}))
                    time.sleep(2)
                    retry_response = guarded_create_completion(model_name, retry_config, imagepath)
                    if retry_response.status_code == 429:
                        time.sleep(10)
                        retry_response = guarded_create_completion(model_name, retry_config, imagepath)
                    if retry_response.status_code >= 500:
                        continue
                    if retry_response.status_code != 200:
                        continue
                    retry_response_json = retry_response.json()
                    if "provider" in retry_response_json:
                        local_providers.add(retry_response_json["provider"])
                    if "choices" not in retry_response_json or not retry_response_json["choices"]:
                        continue
                    retry_message = retry_response_json["choices"][0]["message"]
                    content = retry_message.get("content", "")
                    reasoning_text = retry_message.get("reasoning")
                    choice = retry_response_json["choices"][0]
                    response_json = retry_response_json
                    update_repetition_metrics(content, reasoning_text)
                    step_max_tokens = retry_config.get("max_tokens")
                    if isinstance(step_max_tokens, int) and step_max_tokens > current_max_tokens:
                        current_max_tokens = step_max_tokens
                        local_inference_config["max_tokens"] = step_max_tokens
                    json_obj = parse_with_reasoning_fallback(content, reasoning_text, log_image=imagepath)
                    if json_obj is not None:
                        break

            if json_obj is None:
                log_reasoning_trace(imagepath, message, response_json, retry_stage="parse_failure")
                usage = response_json.get("usage", {})
                parse_failure_entry = base_result_entry()
                parse_failure_entry.update({
                    "_parse_failed": True,
                    "_token_usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                })
                if repetition_detected:
                    parse_failure_entry["_repetition_detected"] = True
                    parse_failure_entry["_repetition_score"] = round(repetition_score, 3)
                    local_rep_score = max(local_rep_score, repetition_score)
                local_results.append(parse_failure_entry)
                status_msg = "[red]Parse failed[/red]"
                return finalize_payload({
                    "imagepath": imagepath,
                    "results": local_results,
                    "providers": list(local_providers),
                    "rep_score": local_rep_score,
                    "status_msg": status_msg,
                    "success": False,
                })

            normalized_obj, schema_errors = extraction_adapter.normalize_output(json_obj)
            schema_retry_attempts = 0

            while schema_errors and schema_retry_attempts < schema_retry_max:
                schema_retry_attempts += 1
                retry_prompt = extraction_adapter.build_schema_retry_prompt(
                    normalized_obj if normalized_obj is not None else json_obj,
                    schema_errors,
                    attempt_number=schema_retry_attempts,
                )
                time.sleep(2)
                retry_response = guarded_create_completion(
                    model_name,
                    local_inference_config,
                    imagepath,
                    prompt_text=retry_prompt,
                )
                if retry_response.status_code == 429:
                    time.sleep(10)
                    retry_response = guarded_create_completion(
                        model_name,
                        local_inference_config,
                        imagepath,
                        prompt_text=retry_prompt,
                    )
                if retry_response.status_code >= 500:
                    continue
                if retry_response.status_code != 200:
                    continue

                retry_response_json = retry_response.json()
                if "provider" in retry_response_json:
                    local_providers.add(retry_response_json["provider"])
                if "choices" not in retry_response_json or not retry_response_json["choices"]:
                    continue

                retry_message = retry_response_json["choices"][0]["message"]
                content = retry_message.get("content", "")
                reasoning_text = retry_message.get("reasoning")
                response_json = retry_response_json
                update_repetition_metrics(content, reasoning_text)

                retry_json_obj = parse_with_reasoning_fallback(content, reasoning_text, log_image=imagepath)
                if retry_json_obj is None:
                    normalized_obj = None
                    schema_errors = ["Retry response could not be parsed as JSON."]
                    continue

                normalized_obj, schema_errors = extraction_adapter.normalize_output(retry_json_obj)
                json_obj = retry_json_obj

            if schema_errors:
                usage = response_json.get("usage", {})
                generation_id = response_json.get("id")

                if not effective_strict_schema:
                    coerced_obj, coercions = extraction_adapter.coerce_invalid_output(normalized_obj)
                    if isinstance(coerced_obj, dict) and coercions:
                        coerced_obj["_schema_coerced"] = True
                        coerced_obj["_schema_errors"] = schema_errors
                        coerced_obj["_schema_retry_attempts"] = schema_retry_attempts
                        coerced_obj["_schema_corrections"] = coercions
                        coerced_obj["_token_usage"] = {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0),
                            "generation_id": generation_id,
                        }
                        if repetition_detected:
                            coerced_obj["_repetition_detected"] = True
                            coerced_obj["_repetition_score"] = round(repetition_score, 3)
                            local_rep_score = max(local_rep_score, repetition_score)
                        local_results.append(coerced_obj)
                        status_msg = extraction_adapter.format_success_status(
                            coerced_obj,
                            schema_coerced=True,
                        )
                        return finalize_payload({
                            "imagepath": imagepath,
                            "results": local_results,
                            "providers": list(local_providers),
                            "rep_score": local_rep_score,
                            "status_msg": status_msg,
                            "success": True,
                        })

                if not effective_strict_schema and isinstance(normalized_obj, dict):
                    schema_passthrough_entry = dict(normalized_obj)
                    schema_passthrough_entry["_schema_failed"] = True
                    schema_passthrough_entry["_schema_passthrough"] = True
                    schema_passthrough_entry["_schema_errors"] = schema_errors
                    schema_passthrough_entry["_schema_retry_attempts"] = schema_retry_attempts
                    schema_passthrough_entry["_token_usage"] = {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                        "generation_id": generation_id,
                    }
                    if repetition_detected:
                        schema_passthrough_entry["_repetition_detected"] = True
                        schema_passthrough_entry["_repetition_score"] = round(repetition_score, 3)
                        local_rep_score = max(local_rep_score, repetition_score)
                    local_results.append(schema_passthrough_entry)
                    status_msg = "[yellow]Schema mismatch (passthrough)[/yellow]"
                    return finalize_payload({
                        "imagepath": imagepath,
                        "results": local_results,
                        "providers": list(local_providers),
                        "rep_score": local_rep_score,
                        "status_msg": status_msg,
                        "success": True,
                    })

                schema_message = response_json.get("choices", [{}])[0].get("message")
                if not isinstance(schema_message, dict):
                    schema_message = {"content": str(schema_message or "")}
                log_reasoning_trace(
                    imagepath,
                    cast(Dict[str, Any], schema_message),
                    response_json,
                    retry_stage="schema_failure",
                )

                schema_failure_entry: Dict[str, Any]
                if isinstance(normalized_obj, dict):
                    schema_failure_entry = dict(normalized_obj)
                else:
                    schema_failure_entry = base_result_entry()
                schema_failure_entry["_schema_failed"] = True
                schema_failure_entry["_schema_errors"] = schema_errors
                schema_failure_entry["_schema_retry_attempts"] = schema_retry_attempts
                schema_failure_entry["_token_usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "generation_id": generation_id,
                }
                if repetition_detected:
                    schema_failure_entry["_repetition_detected"] = True
                    schema_failure_entry["_repetition_score"] = round(repetition_score, 3)
                    local_rep_score = max(local_rep_score, repetition_score)
                local_results.append(schema_failure_entry)
                status_msg = "[red]Schema failed[/red]"
                return finalize_payload({
                    "imagepath": imagepath,
                    "results": local_results,
                    "providers": list(local_providers),
                    "rep_score": local_rep_score,
                    "status_msg": status_msg,
                    "success": False,
                })

            if not isinstance(normalized_obj, dict):
                normalized_obj = base_result_entry()
            json_obj = normalized_obj

            usage = response_json.get("usage", {})
            generation_id = response_json.get("id")
            json_obj["_token_usage"] = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
                "generation_id": generation_id
            }
            if repetition_detected:
                json_obj["_repetition_detected"] = True
                json_obj["_repetition_score"] = round(repetition_score, 3)
                local_rep_score = max(local_rep_score, repetition_score)
            local_results.append(json_obj)

            status_msg = extraction_adapter.format_success_status(json_obj)

            return finalize_payload({
                "imagepath": imagepath,
                "results": local_results,
                "providers": list(local_providers),
                "rep_score": local_rep_score,
                "status_msg": status_msg,
                "success": True,
            })

        except Exception as e:
            exception_entry = base_result_entry()
            exception_entry.update({
                "_exception": str(e)
            })
            local_results.append(exception_entry)
            return finalize_payload({
                "imagepath": imagepath,
                "results": local_results,
                "providers": list(local_providers),
                "rep_score": local_rep_score,
                "status_msg": f"[red]Error: {str(e)[:20]}...[/red]",
                "success": False,
            })

    # Create progress bar with live status, then dispatch tasks
    with create_inference_progress() as progress:
        task = add_inference_task(progress, total_images)

        critical_stop: Optional[tuple[int, str]] = None
        with ThreadPoolExecutor(max_workers=max(1, concurrency)) as executor:
            future_map = {executor.submit(process_image, img): img for img in imagepaths}
            for future in as_completed(future_map):
                result_payload = future.result()
                if "critical_error" in result_payload:
                    code, msg = result_payload["critical_error"]
                    progress.update(task, last_result=f"[red]{msg} - stopping[/red]")
                    critical_stop = (code, msg)
                    break

                img = result_payload["imagepath"]
                local_results = result_payload["results"]
                local_providers = result_payload["providers"]
                local_rep_score = result_payload["rep_score"]
                status_msg = result_payload["status_msg"]
                success = result_payload["success"]

                # Merge providers
                with provider_lock:
                    for p in local_providers:
                        config.additional_config["actual_model_providers"].add(p)
                if local_rep_score:
                    current_best = repetition_event_scores.get(img, 0.0)
                    if local_rep_score > current_best:
                        repetition_event_scores[img] = local_rep_score

                results[img].extend(local_results)
                completed_images += 1
                if success:
                    successful_images += 1
                else:
                    failed_images.append((img, summarize_failure_reason(local_results)))

                # Save incrementally
                manager.save_results(config.run_name, dict(results))

                success_rate = f"{(successful_images/max(1, completed_images))*100:.1f}%"
                progress.update(
                    task,
                    advance=1,
                    success_rate=success_rate,
                    last_result=status_msg[:40] + "..." if len(status_msg) > 40 else status_msg,
                )

            # If we hit a critical stop, cancel remaining futures
            if critical_stop is not None:
                for pending_future in future_map:
                    pending_future.cancel()
                console.print(f"\n[red]❌ Critical Error: {critical_stop[1]} ({critical_stop[0]})[/red]")
                console.print("[yellow]⚠️  Processing stopped. Please resolve the issue and rerun.[/yellow]")
    
    if repetition_event_scores:
        worst_image, worst_score = max(repetition_event_scores.items(), key=lambda item: item[1])
        console.print(
            f"\n[yellow]⚠️ Detected repetitive output in {len(repetition_event_scores)} responses (max score {worst_score:.2f}).[/yellow]"
        )
        console.print(
            f"[dim]Example image: {worst_image}. Consider increasing repetition penalty via --repetition-penalty when rerunning.[/dim]"
        )

    # Calculate actual runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    runtime_formatted = format_runtime(runtime_seconds)
    
    # Update config with actual runtime
    # Create config dict and add extracted model information
    config_dict = config.to_dict()
    config_dict["environment"]["runtime"] = runtime_formatted
    config_dict["additional"]["actual_runtime_seconds"] = runtime_seconds
    config_dict["api"]["repetition_penalty"] = effective_repetition_penalty
    
    # Convert provider set to list for serialization
    if "actual_model_providers" in config_dict["additional"]:
        config_dict["additional"]["actual_model_providers"] = sorted(list(config_dict["additional"]["actual_model_providers"]))
    
    # Add extracted model information to config (avoid duplication)
    if supported_parameters:
        # Enhance supported parameters with OpenRouter default values
        param_defaults = {
            "temperature": 1.0,  # OpenRouter default
            "top_p": 1.0,        # OpenRouter default  
            "max_tokens": None,   # No default, model-dependent
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "repetition_penalty": 1.0,
            "top_k": None,        # No default
            "min_p": None,        # No default
            "seed": None,         # No default
            "stop": None,         # No default
            "response_format": None,  # No default
            "tool_choice": "auto",
            "logit_bias": None,   # No default
            "include_reasoning": None,  # Model-specific
            "reasoning": None     # Model-specific
        }
        
        # Create enhanced parameter info with defaults
        enhanced_params = {}
        for param in supported_parameters:
            enhanced_params[param] = {
                "supported": True,
                "openrouter_default": param_defaults.get(param, "Unknown")
            }
        
        config_dict["additional"]["model_supported_parameters"] = enhanced_params
        
    if model_context_length != "Unknown":
        config_dict["additional"]["model_context_length"] = model_context_length
    if model_pricing:
        config_dict["additional"]["model_pricing"] = model_pricing
    
    # Save updated config and refresh manifest metadata
    manager.save_run_config(config.run_name, config_dict)
    
    # Final results save (incremental saves were done during processing)
    results_dict = dict(results)
    manager.save_results(config.run_name, results_dict)

    schema_failed_images: list[tuple[str, list[str], int]] = []
    schema_coerced_images: list[tuple[str, list[str], int]] = []
    for imagepath, result_list in results_dict.items():
        if not result_list or not isinstance(result_list[0], dict):
            continue
        entry = result_list[0]
        retry_attempts_raw = entry.get("_schema_retry_attempts", 0)
        retry_attempts = int(retry_attempts_raw) if isinstance(retry_attempts_raw, (int, float)) else 0

        if entry.get("_schema_failed"):
            errors_raw = entry.get("_schema_errors")
            errors = [str(err) for err in errors_raw] if isinstance(errors_raw, list) else []
            schema_failed_images.append((imagepath, errors, retry_attempts))
        elif entry.get("_schema_coerced"):
            corrections_raw = entry.get("_schema_corrections")
            corrections = [str(c) for c in corrections_raw] if isinstance(corrections_raw, list) else []
            schema_coerced_images.append((imagepath, corrections, retry_attempts))
    
    # Display results with rich formatting
    console.print("\n[bold green]✅ Inference Complete![/bold green]")
    
    from rich.table import Table
    results_table = Table(show_header=False, box=None)
    failed_count = total_images - successful_images
    results_table.add_row("[cyan]Successful:[/cyan]", f"{successful_images}/{total_images} images")
    results_table.add_row("[cyan]Failed:[/cyan]", f"{failed_count}/{total_images} images")
    results_table.add_row("[cyan]Success rate:[/cyan]", f"{successful_images/total_images*100:.1f}%")
    results_table.add_row("[cyan]Runtime:[/cyan]", f"[bold]{runtime_formatted}[/bold]")
    results_table.add_row("[cyan]API Router:[/cyan]", "OpenRouter")
    results_table.add_row("[cyan]Extraction mode:[/cyan]", extraction_spec.mode)
    results_table.add_row("[cyan]Strict schema:[/cyan]", str(effective_strict_schema))
    if provider_routing_effective:
        results_table.add_row("[cyan]Routing config:[/cyan]", json.dumps(provider_routing_effective, sort_keys=True))
    else:
        results_table.add_row("[cyan]Routing config:[/cyan]", "OpenRouter default")
    
    # Show actual model providers used
    actual_providers = sorted(list(config.additional_config.get("actual_model_providers", set())))
    if actual_providers:
        results_table.add_row("[cyan]Model Providers:[/cyan]", ", ".join(actual_providers))
    if schema_failed_images:
        results_table.add_row("[yellow]Schema failures:[/yellow]", f"{len(schema_failed_images)}/{total_images} images")
    if schema_coerced_images:
        results_table.add_row("[yellow]Schema coerced:[/yellow]", f"{len(schema_coerced_images)}/{total_images} images")
    
    results_table.add_row("[cyan]Results saved:[/cyan]", f"{run_dir}/results.json")
    results_table.add_row("[cyan]Run name:[/cyan]", f"[bold]{config.run_name}[/bold]")
    
    console.print(results_table)

    if schema_failed_images:
        console.print(
            f"\n[yellow]⚠️ Schema validation failed for {len(schema_failed_images)} image(s) after retries.[/yellow]"
        )

    if schema_coerced_images:
        console.print(
            f"\n[yellow]⚠️ Schema coercion applied for {len(schema_coerced_images)} image(s) "
            "(invalid fields were set to empty string).[/yellow]"
        )
        from rich.table import Table as RichTable
        coercion_table = RichTable(show_header=True, header_style="bold yellow")
        coercion_table.add_column("Image", style="cyan")
        coercion_table.add_column("Corrections", style="white")
        coercion_table.add_column("Retries", style="magenta")
        max_rows = 8
        for imagepath, corrections, retries in schema_coerced_images[:max_rows]:
            correction_text = "; ".join(corrections) if corrections else "field coercion applied"
            coercion_table.add_row(Path(imagepath).name, correction_text, str(retries))
        console.print(coercion_table)
        remaining = len(schema_coerced_images) - max_rows
        if remaining > 0:
            console.print(f"[dim]... and {remaining} more coerced image(s).[/dim]")

    if failed_images:
        console.print("\n[yellow]⚠️ Failed image summary:[/yellow]")
        failure_table = Table(show_header=True, header_style="bold yellow")
        failure_table.add_column("Image", style="cyan")
        failure_table.add_column("Reason", style="white")
        max_rows = 12
        for imagepath, reason in failed_images[:max_rows]:
            failure_table.add_row(Path(imagepath).name, reason)
        console.print(failure_table)
        remaining = len(failed_images) - max_rows
        if remaining > 0:
            console.print(f"[dim]... and {remaining} more failed image(s).[/dim]")
    
    # After inference is complete, batch process generation IDs for precise costs
    console.print("\n[cyan]🔍 Fetching precise cost data...[/cyan]")
    original_results = dict(results)
    updated_results = batch_update_generation_costs(
        original_results,
        max_workers=max(1, concurrency),
    )
    
    # Always save the updated results (includes cost fetch metadata for any unrecovered failures)
    manager.save_results(config.run_name, updated_results)

    external_output_path: Optional[str] = None
    if output_json:
        try:
            output_path = Path(output_json).expanduser().resolve(strict=False)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(updated_results, f, indent=2, ensure_ascii=False)
            external_output_path = str(output_path)
        except Exception as exc:
            console.print(f"[yellow]⚠️ Failed to write --output-json file: {exc}[/yellow]")
    
    # Count how many actually got precise costs
    cost_count = 0
    for result_list in updated_results.values():
        if result_list and len(result_list) > 0:
            result = result_list[0]
            token_usage = result.get("_token_usage", {})
            if "actual_cost" in token_usage:
                cost_count += 1
    
    if cost_count > 0:
        console.print(f"[green]✅ Added precise costs to {cost_count} results[/green]")
    else:
        console.print("[yellow]⚠️ No precise costs could be retrieved[/yellow]")

    if external_output_path:
        console.print(f"[green]✅ Wrote output JSON copy to {external_output_path}[/green]")
    
    return config.run_name


if __name__ == "__main__":
    from .cli import main
    main()
