#!/usr/bin/env python3
"""
OpenRouter inference engine and orchestration.
"""
import time
import yaml  # type: ignore[import]
import re
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Optional, cast

from rich.console import Console

from ..core import RunManager, RunConfig, format_runtime, create_inference_progress, add_inference_task, get_imagepaths
from .api import (
    MODEL_CONFIG_OVERRIDES, 
    create_completion, 
    parse_response_content, 
    fetch_openrouter_models,
    batch_update_generation_costs,
)
from ..core.prompts import STUDENT_EXTRACTION_PROMPT
from .ui import interactive_config_prompt

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
                            temperature: float = 0.0,
                            max_tokens: int = 4096,
                            top_p: float = 1.0,
                            repetition_penalty: Optional[float] = None,
                            model_size: Optional[str] = None,
                            open_weights: Optional[bool] = None,
                            license_info: Optional[str] = None,
                            interactive: bool = False,
                            concurrency: int = 1,
                            rate_limit: Optional[float] = None):
    """Run inference using any OpenRouter vision model."""
    
    # Start timing
    start_time = time.time()

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
    
    # Parse model name
    if "/" in model_name:
        org, model = model_name.split("/", 1)
    else:
        org, model = "unknown", model_name
    
    # Get model-specific overrides if they exist
    overrides = cast(Dict[str, Any], MODEL_CONFIG_OVERRIDES.get(model_name, {}))

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
            "top_p": top_p,
            "repetition_penalty": effective_repetition_penalty,
            "prompt_template": STUDENT_EXTRACTION_PROMPT,
            "actual_model_providers": set(),  # Will track actual model providers used
        }
    )
    
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
    
    # Setup inference parameters
    pages = [1, 3]
    folder = "imgs/q11/"
    pattern = r"doc-\d+-page-[" + "".join([str(p) for p in pages]) + "]-[A-Z0-9]+.png"
    imagepaths = get_imagepaths(folder, pattern)
    
    inference_config: Dict[str, Any] = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }

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

    def guarded_create_completion(local_model_name: str, local_config: Dict[str, Any], img_path: str):
        rate_limiter.acquire()
        return create_completion(local_model_name, local_config, img_path)

    # Run inference with rich progress tracking
    results = defaultdict(list)
    total_images = len(imagepaths)
    completed_images = 0
    successful_images = 0
    repetition_event_scores: Dict[str, float] = {}
    provider_lock = threading.Lock()

    # Per-image worker encapsulating the existing logic. Avoids disk writes and progress updates.
    def process_image(imagepath: str):
        local_results: list[Dict[str, Any]] = []
        local_providers: set[str] = set()
        local_rep_score: float = 0.0
        status_msg: str = ""

        # Copy base inference config for this image
        local_inference_config: Dict[str, Any] = dict(inference_config)
        try:
            response = guarded_create_completion(model_name, local_inference_config, imagepath)

            if response.status_code == 429:
                time.sleep(10)
                response = guarded_create_completion(model_name, local_inference_config, imagepath)

            if response.status_code == 402:
                return {
                    "critical_error": (402, "Insufficient funds"),
                    "imagepath": imagepath,
                }
            if response.status_code == 401:
                return {
                    "critical_error": (401, "Invalid API key"),
                    "imagepath": imagepath,
                }
            if response.status_code >= 500:
                return {
                    "critical_error": (response.status_code, "Server error"),
                    "imagepath": imagepath,
                }
            if response.status_code >= 400:
                try:
                    error_usage = response.json().get("usage", {})
                except Exception:
                    error_usage = {}
                local_results.append({
                    "student_full_name": "",
                    "university_id": "",
                    "section_number": "",
                    "_api_error": response.status_code,
                    "_token_usage": {
                        "prompt_tokens": error_usage.get("prompt_tokens", 0),
                        "completion_tokens": error_usage.get("completion_tokens", 0),
                        "total_tokens": error_usage.get("total_tokens", 0)
                    }
                })
                status_msg = f"[red]API Error {response.status_code}[/red]"
                return {
                    "imagepath": imagepath,
                    "results": local_results,
                    "providers": list(local_providers),
                    "rep_score": local_rep_score,
                    "status_msg": status_msg,
                    "success": False,
                }

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
                    local_results.append({
                        "student_full_name": "",
                        "university_id": "",
                        "section_number": "",
                        "_no_response": True,
                        "_no_response_error": error_details,
                        "_token_usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                    })
                    status_msg = "[red]No response[/red]"
                    return {
                        "imagepath": imagepath,
                        "results": local_results,
                        "providers": list(local_providers),
                        "rep_score": local_rep_score,
                        "status_msg": status_msg,
                        "success": False,
                    }

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
                            local_results.append({
                                "student_full_name": "",
                                "university_id": "",
                                "section_number": "",
                                "_empty_response": True,
                                "_retry_failed": True
                            })
                            status_msg = "[red]Empty after retry[/red]"
                            return {
                                "imagepath": imagepath,
                                "results": local_results,
                                "providers": list(local_providers),
                                "rep_score": local_rep_score,
                                "status_msg": status_msg,
                                "success": False,
                            }
                else:
                    local_results.append({
                        "student_full_name": "",
                        "university_id": "",
                        "section_number": "",
                        "_empty_response": True,
                        "_retry_failed": True
                    })
                    status_msg = "[red]Retry failed[/red]"
                    return {
                        "imagepath": imagepath,
                        "results": local_results,
                        "providers": list(local_providers),
                        "rep_score": local_rep_score,
                        "status_msg": status_msg,
                        "success": False,
                    }

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
                choice = retry_response_json["choices"][0]
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
                parse_failure_entry = {
                    "student_full_name": "",
                    "university_id": "",
                    "section_number": "",
                    "_parse_failed": True,
                    "_token_usage": {
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0)
                    }
                }
                if repetition_detected:
                    parse_failure_entry["_repetition_detected"] = True
                    parse_failure_entry["_repetition_score"] = round(repetition_score, 3)
                    local_rep_score = max(local_rep_score, repetition_score)
                local_results.append(parse_failure_entry)
                status_msg = "[red]Parse failed[/red]"
                return {
                    "imagepath": imagepath,
                    "results": local_results,
                    "providers": list(local_providers),
                    "rep_score": local_rep_score,
                    "status_msg": status_msg,
                    "success": False,
                }

            if "ufid" in json_obj:
                json_obj["university_id"] = json_obj.pop("ufid", "")
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

            name = json_obj.get('student_full_name', 'N/A')
            ufid = json_obj.get('university_id', '')
            status_msg = f"✓ {name} (ID: {ufid})" if ufid else f"✓ {name}"

            return {
                "imagepath": imagepath,
                "results": local_results,
                "providers": list(local_providers),
                "rep_score": local_rep_score,
                "status_msg": status_msg,
                "success": True,
            }

        except Exception as e:
            local_results.append({
                "student_full_name": "",
                "university_id": "",
                "section_number": "",
                "_exception": str(e)
            })
            return {
                "imagepath": imagepath,
                "results": local_results,
                "providers": list(local_providers),
                "rep_score": local_rep_score,
                "status_msg": f"[red]Error: {str(e)[:20]}...[/red]",
                "success": False,
            }

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
    
    # Save updated config
    config_path = Path(run_dir) / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    # Final results save (incremental saves were done during processing)
    results_dict = dict(results)
    manager.save_results(config.run_name, results_dict)
    
    # Display results with rich formatting
    console.print("\n[bold green]✅ Inference Complete![/bold green]")
    
    from rich.table import Table
    results_table = Table(show_header=False, box=None)
    results_table.add_row("[cyan]Processed:[/cyan]", f"{successful_images}/{total_images} images")
    results_table.add_row("[cyan]Success rate:[/cyan]", f"{successful_images/total_images*100:.1f}%")
    results_table.add_row("[cyan]Runtime:[/cyan]", f"[bold]{runtime_formatted}[/bold]")
    results_table.add_row("[cyan]API Router:[/cyan]", "OpenRouter")
    
    # Show actual model providers used
    actual_providers = sorted(list(config.additional_config.get("actual_model_providers", set())))
    if actual_providers:
        results_table.add_row("[cyan]Model Providers:[/cyan]", ", ".join(actual_providers))
    
    results_table.add_row("[cyan]Results saved:[/cyan]", f"{run_dir}/results.json")
    results_table.add_row("[cyan]Run name:[/cyan]", f"[bold]{config.run_name}[/bold]")
    
    console.print(results_table)
    
    # After inference is complete, batch process generation IDs for precise costs
    console.print("\n[cyan]🔍 Fetching precise cost data...[/cyan]")
    original_results = dict(results)
    updated_results = batch_update_generation_costs(original_results)
    
    # Always save the updated results (generation IDs should be removed even if no costs added)
    manager.save_results(config.run_name, updated_results)
    
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
    
    return config.run_name


if __name__ == "__main__":
    from .cli import main
    main()