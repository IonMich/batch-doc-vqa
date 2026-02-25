#!/usr/bin/env python3
"""
OpenRouter API interactions and data processing utilities.
"""
import os
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests  # type: ignore[import]
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from .presets import resolve_preset_definition
from ..core import filepath_to_base64

console = Console()

# Qwen 3.5 recommended generation params for thinking/coding workloads.
QWEN_35_RECOMMENDED_DEFAULTS = {
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
    "presence_penalty": 0.0,
    "repetition_penalty": 1.0,
}

# Model-specific configurations for special cases
MODEL_CONFIG_OVERRIDES = {
    "z-ai/glm-4.5v": {
        "response_format": "box",  # Uses <|begin_of_box|> format
    },
    "qwen/qwen3.5-35b-a3b": dict(QWEN_35_RECOMMENDED_DEFAULTS),
    "qwen/qwen3.5-27b": dict(QWEN_35_RECOMMENDED_DEFAULTS),
    "qwen/qwen3.5-122b-a10b": dict(QWEN_35_RECOMMENDED_DEFAULTS),
    "qwen/qwen3.5-flash-02-23": dict(QWEN_35_RECOMMENDED_DEFAULTS),
    "qwen/qwen3.5-397b-a17b": dict(QWEN_35_RECOMMENDED_DEFAULTS),
    "qwen/qwen3.5-plus-02-15": {
        # Qwen provider "thinking mode" defaults.
        **QWEN_35_RECOMMENDED_DEFAULTS,
    },
    # Add more overrides as needed for models with special requirements
}


def resolve_model_config_overrides(model_name: str) -> Dict[str, Any]:
    """Resolve model overrides with light normalization for variant suffixes."""
    normalized = (model_name or "").strip().lower()
    if not normalized:
        return {}

    candidates = [normalized]
    base_model_name = normalized.split(":", 1)[0]
    if base_model_name not in candidates:
        candidates.append(base_model_name)

    for candidate in candidates:
        overrides = MODEL_CONFIG_OVERRIDES.get(candidate)
        if isinstance(overrides, dict):
            return dict(overrides)

    return {}


def create_completion(
    model_name: str,
    config: Dict[str, Any],
    imagepath: str,
    *,
    prompt_text: Optional[str] = None,
):
    """Create a completion request to OpenRouter."""
    request_config = dict(config)
    # Internal-only key (if present) should not be forwarded to the API payload.
    template_from_config = request_config.pop("prompt_template", None)

    if prompt_text is None:
        if isinstance(template_from_config, str) and template_from_config.strip():
            prompt_text = template_from_config
        else:
            prompt_text = resolve_preset_definition().prompt_text

    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        },
        json={
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_text
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{filepath_to_base64(imagepath)}",
                            },
                        },
                    ],
                }
            ],
            **request_config,
        },
        timeout=180,
    )
    
    return response


def parse_response_content(content: str, response_format: str) -> Optional[Dict[str, Any]]:
    """Parse the response content based on the model's response format."""
    if not content or content.strip() == "":
        return None
    
    # Remove thinking tags and content - handle various thinking formats
    import re
    thinking_patterns = [
        r'◁think▷.*?◁/think▷',  # ◁think▷...◁/think▷
        r'<think>.*?</think>',   # <think>...</think>  
        r'<thinking>.*?</thinking>',  # <thinking>...</thinking>
        r'◁think▷.*?(?=\{|So my JSON|JSON response:|Response:)',  # ◁think▷ until JSON starts
    ]
    
    for pattern in thinking_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    content = content.strip()
    
    if response_format == "box":
        # Handle models that wrap JSON in special tokens (like GLM)
        if "<|begin_of_box|>" in content and "<|end_of_box|>" in content:
            start = content.find("<|begin_of_box|>") + len("<|begin_of_box|>")
            end = content.find("<|end_of_box|>")
            content = content[start:end].strip()
    
    # Try to parse JSON directly
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # If direct parsing fails, try to extract JSON from markdown
        import re
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Generic fallback: scan for first decodable JSON object in the text.
        decoder = json.JSONDecoder()
        for idx, char in enumerate(content):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(content[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                return candidate
                
        return None


def fetch_openrouter_providers() -> Optional[List[Dict[str, Any]]]:
    """Fetch available providers from OpenRouter API."""
    try:
        response = requests.get("https://openrouter.ai/api/v1/providers")
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        console.print(f"[red]Error fetching providers: {e}[/red]")
        return None


def fetch_openrouter_models() -> Optional[List[Dict[str, Any]]]:
    """Fetch available models from OpenRouter API."""
    try:
        response = requests.get("https://openrouter.ai/api/v1/models")
        response.raise_for_status()
        return response.json().get("data", [])
    except Exception as e:
        console.print(f"[red]Error fetching models: {e}[/red]")
        return None


def model_supports_image_input(model: Dict[str, Any]) -> bool:
    """Return whether a model is image-capable based on architecture metadata."""
    architecture = model.get("architecture")
    if not isinstance(architecture, dict):
        return False

    input_modalities = architecture.get("input_modalities")
    if isinstance(input_modalities, list):
        for modality in input_modalities:
            if not isinstance(modality, str):
                continue
            token = modality.strip().lower()
            if "image" in token or "vision" in token:
                return True

    modality = architecture.get("modality")
    if isinstance(modality, str):
        lowered = modality.strip().lower()
        if "image" in lowered or "vision" in lowered:
            return True

    return False


def filter_vision_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter models to only include image-capable models."""
    return [model for model in models if model_supports_image_input(model)]


_RETRYABLE_GENERATION_STATUS_CODES = {404, 408, 409, 425, 429}


def _env_flag(name: str, default: bool = False) -> bool:
    """Parse boolean-like environment variable values."""
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    return default


def _copy_scalar_field(
    target: Dict[str, Any],
    source: Dict[str, Any],
    source_key: str,
    *,
    target_key: Optional[str] = None,
) -> None:
    """Copy scalar value from source into target when present and non-empty."""
    value = source.get(source_key)
    if value is None:
        return
    if isinstance(value, str):
        trimmed = value.strip()
        if not trimmed:
            return
        target[target_key or source_key] = trimmed
        return
    if isinstance(value, (int, float, bool)):
        target[target_key or source_key] = value


def _extract_generation_meta(
    data: Dict[str, Any],
    *,
    generation_id: Optional[str] = None,
    include_generation_id: bool = False,
) -> Optional[Dict[str, Any]]:
    """Extract a conservative generation metadata subset safe for result artifacts."""
    if not isinstance(data, dict):
        return None

    meta: Dict[str, Any] = {}

    scalar_fields = [
        "provider",
        "model",
        "status",
        "route",
        "finish_reason",
        "native_finish_reason",
        "service_tier",
        "cache_status",
    ]
    numeric_fields = [
        "latency",
        "latency_ms",
        "duration",
        "duration_ms",
        "processing_time",
        "processing_time_ms",
        "queue_time",
        "queue_time_ms",
        "time_to_first_token",
        "time_to_first_token_ms",
    ]
    timestamp_fields = [
        "created_at",
        "generated_at",
        "finished_at",
        "created_at_ms",
        "generated_at_ms",
        "finished_at_ms",
    ]

    for key in scalar_fields:
        _copy_scalar_field(meta, data, key)

    for key in numeric_fields:
        _copy_scalar_field(meta, data, key)

    for key in timestamp_fields:
        _copy_scalar_field(meta, data, key)

    # Some payloads place provider details under an object.
    provider_obj = data.get("provider")
    if isinstance(provider_obj, dict):
        _copy_scalar_field(meta, provider_obj, "name", target_key="provider_name")
        _copy_scalar_field(meta, provider_obj, "slug", target_key="provider_slug")
        if "provider" not in meta:
            provider_slug = meta.get("provider_slug")
            provider_name = meta.get("provider_name")
            if isinstance(provider_slug, str):
                meta["provider"] = provider_slug
            elif isinstance(provider_name, str):
                meta["provider"] = provider_name

    if include_generation_id and isinstance(generation_id, str) and generation_id.strip():
        meta["generation_id"] = generation_id.strip()

    if not meta:
        return None
    return meta


def _fetch_generation_stats_with_retries(
    generation_id: str,
    api_key: str,
    *,
    max_retries: int = 3,
    base_delay_seconds: float = 0.75,
    request_timeout_seconds: float = 5.0,
) -> Dict[str, Any]:
    """Fetch generation stats with bounded retries/backoff."""
    attempts = 0
    last_status_code: Optional[int] = None
    last_error: Optional[str] = None
    last_retryable = False

    max_retries = max(0, int(max_retries))
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        attempts = attempt
        try:
            response = requests.get(
                f"https://openrouter.ai/api/v1/generation?id={generation_id}",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                timeout=request_timeout_seconds,
            )
            last_status_code = response.status_code

            if response.status_code == 200:
                try:
                    stats_data = response.json()
                except Exception as exc:
                    last_error = f"invalid json: {type(exc).__name__}: {exc}"
                    last_retryable = True
                else:
                    data = stats_data.get("data", {})
                    if data:
                        return {
                            "success": True,
                            "attempts": attempts,
                            "status_code": response.status_code,
                            "retryable": False,
                            "error": None,
                            "data": data,
                        }
                    last_error = "empty data payload"
                    last_retryable = True
            else:
                last_error = f"HTTP {response.status_code}"
                last_retryable = (
                    response.status_code in _RETRYABLE_GENERATION_STATUS_CODES
                    or response.status_code >= 500
                )
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            last_retryable = True

        if not last_retryable or attempt >= total_attempts:
            break

        backoff_seconds = min(10.0, base_delay_seconds * (2 ** (attempt - 1)))
        time.sleep(backoff_seconds)

    return {
        "success": False,
        "attempts": attempts,
        "status_code": last_status_code,
        "retryable": last_retryable,
        "error": last_error,
        "data": None,
    }


def batch_update_generation_costs(results: Dict, *, max_workers: int = 1) -> Dict:
    """Update results with precise costs from generation API in batch."""
    
    # Collect all generation IDs that need processing
    generation_ids_to_process = []
    for filepath, result_list in results.items():
        if result_list and len(result_list) > 0:
            result = result_list[0]
            token_usage = result.get("_token_usage", {})
            generation_id = token_usage.get("generation_id")
            if generation_id and "actual_cost" not in token_usage:
                generation_ids_to_process.append((filepath, generation_id))
    
    if not generation_ids_to_process:
        return results
    
    # Wait a bit for all generation stats to be ready
    console.print(f"[dim]Found {len(generation_ids_to_process)} generation IDs to process[/dim]")
    console.print("[dim]Waiting 3 seconds for generation stats to be ready...[/dim]")
    time.sleep(3)
    
    # Process generation IDs in batch
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        console.print("[yellow]⚠️ No API key found, skipping precise cost updates[/yellow]")
        return results
    
    requested_workers = max(1, int(max_workers))
    worker_count = min(requested_workers, len(generation_ids_to_process))
    max_retries = 3
    base_delay_seconds = 0.75
    request_timeout_seconds = 5.0
    keep_generation_id = _env_flag("OPENROUTER_KEEP_GENERATION_ID", default=False)

    updated_count = 0
    zero_cost_count = 0
    failed_count = 0
    retried_count = 0
    failed_fetches: list[tuple[str, str, Dict[str, Any]]] = []
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        "•",
        TimeElapsedColumn(),
        console=console,
        transient=True
    ) as progress:
        
        task = progress.add_task(
            f"Updating costs ({worker_count} workers)...",
            total=len(generation_ids_to_process),
        )

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_map = {
                executor.submit(
                    _fetch_generation_stats_with_retries,
                    generation_id,
                    api_key,
                    max_retries=max_retries,
                    base_delay_seconds=base_delay_seconds,
                    request_timeout_seconds=request_timeout_seconds,
                ): (filepath, generation_id)
                for filepath, generation_id in generation_ids_to_process
            }

            for future in as_completed(future_map):
                filepath, generation_id = future_map[future]
                result = results[filepath][0]
                token_usage = result.setdefault("_token_usage", {})

                try:
                    fetch_outcome = future.result()
                except Exception as exc:
                    fetch_outcome = {
                        "success": False,
                        "attempts": 1,
                        "status_code": None,
                        "retryable": True,
                        "error": f"worker exception: {type(exc).__name__}: {exc}",
                        "data": None,
                    }

                attempts = int(fetch_outcome.get("attempts", 1) or 1)
                if attempts > 1:
                    retried_count += 1

                if fetch_outcome.get("success"):
                    data = fetch_outcome.get("data") or {}
                    if "native_tokens_prompt" in data:
                        token_usage["prompt_tokens"] = data["native_tokens_prompt"]
                    if "native_tokens_completion" in data:
                        token_usage["completion_tokens"] = data["native_tokens_completion"]

                    actual_cost = data.get("total_cost")
                    if isinstance(actual_cost, (int, float)):
                        token_usage["actual_cost"] = float(actual_cost)
                        if float(actual_cost) > 0:
                            updated_count += 1
                        else:
                            zero_cost_count += 1
                    else:
                        zero_cost_count += 1

                    generation_meta = _extract_generation_meta(
                        data,
                        generation_id=generation_id,
                        include_generation_id=keep_generation_id,
                    )
                    if generation_meta:
                        result["_generation_meta"] = generation_meta

                    if not keep_generation_id:
                        token_usage.pop("generation_id", None)
                    result.pop("_cost_fetch", None)
                else:
                    failed_count += 1
                    failure_meta = {
                        "status": "failed",
                        "attempts": attempts,
                        "status_code": fetch_outcome.get("status_code"),
                        "retryable": bool(fetch_outcome.get("retryable")),
                        "error": fetch_outcome.get("error"),
                    }
                    if keep_generation_id:
                        failure_meta["generation_id"] = generation_id
                    else:
                        token_usage.pop("generation_id", None)
                    result["_cost_fetch"] = failure_meta
                    failed_fetches.append((filepath, generation_id, failure_meta))

                progress.update(
                    task,
                    advance=1,
                    description=(
                        f"Updated {updated_count} costs • "
                        f"retried {retried_count} • "
                        f"failed {failed_count}"
                    ),
                )

    total_targets = len(generation_ids_to_process)
    if updated_count > 0:
        console.print(
            f"[green]✅ Updated {updated_count}/{total_targets} results with precise costs[/green]"
        )
    else:
        console.print("[yellow]⚠️ Could not get precise costs for any results[/yellow]")

    if zero_cost_count > 0:
        console.print(
            f"[dim]ℹ️ {zero_cost_count}/{total_targets} generation fetches returned zero/unknown cost[/dim]"
        )

    if retried_count > 0:
        console.print(f"[dim]ℹ️ Retried {retried_count} generation lookups[/dim]")

    if failed_fetches:
        console.print(
            f"[yellow]⚠️ Cost fetch failed for {failed_count}/{total_targets} results after retries[/yellow]"
        )
        max_rows = 8
        for filepath, _generation_id, failure_meta in failed_fetches[:max_rows]:
            image_name = os.path.basename(filepath)
            status = failure_meta.get("status_code")
            error = failure_meta.get("error") or "unknown error"
            console.print(f"[dim]- {image_name}: status={status}, error={error}[/dim]")
        remaining = len(failed_fetches) - max_rows
        if remaining > 0:
            console.print(f"[dim]... and {remaining} more failed cost fetches[/dim]")
    
    return results
