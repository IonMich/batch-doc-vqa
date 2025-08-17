#!/usr/bin/env python3
"""
OpenRouter inference engine and orchestration.
"""
import time
import yaml
from collections import defaultdict
from pathlib import Path
from typing import Optional

from rich.console import Console

from ..core import RunManager, RunConfig, format_runtime, create_inference_progress, add_inference_task
from .api import (
    MODEL_CONFIG_OVERRIDES, 
    create_completion, 
    parse_response_content, 
    fetch_openrouter_models,
    batch_update_generation_costs,
)
from ..core.prompts import STUDENT_EXTRACTION_PROMPT
from .ui import interactive_config_prompt
from .cli import get_imagepaths

console = Console()


def run_openrouter_inference(model_name: str, 
                            temperature: float = 0.0,
                            max_tokens: int = 4096,
                            top_p: float = 1.0,
                            model_size: Optional[str] = None,
                            open_weights: Optional[bool] = None,
                            license_info: Optional[str] = None,
                            interactive: bool = False):
    """Run inference using any OpenRouter vision model."""
    
    # Start timing
    start_time = time.time()
    
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
    overrides = MODEL_CONFIG_OVERRIDES.get(model_name, {})
    
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
            "prompt_template": STUDENT_EXTRACTION_PROMPT,
            "actual_model_providers": set(),  # Will track actual model providers used
        }
    )
    
    # Create run directory
    manager = RunManager()
    run_dir = manager.create_run_directory(config)
    
    print(f"Starting OpenRouter inference run: {config.run_name}")
    print(f"Model: {model_name}")
    print(f"Run directory: {run_dir}")
    
    # Setup inference parameters
    pages = [1, 3]
    folder = "imgs/q11/"
    pattern = r"doc-\d+-page-[" + "".join([str(p) for p in pages]) + "]-[A-Z0-9]+.png"
    imagepaths = get_imagepaths(folder, pattern)
    
    inference_config = {
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    
    # Run inference with rich progress tracking
    results = defaultdict(list)
    total_images = len(imagepaths)
    successful_images = 0
    
    # Create progress bar with live status
    with create_inference_progress() as progress:
        task = add_inference_task(progress, total_images)
        
        for i, imagepath in enumerate(imagepaths, 1):            
            try:
                response = create_completion(model_name, inference_config, imagepath)
                
                # Check for rate limiting or other API errors
                if response.status_code == 429:
                    progress.update(task, last_result="[yellow]Rate limited, waiting...[/yellow]")
                    time.sleep(10)
                    # Retry the request
                    response = create_completion(model_name, inference_config, imagepath)
                
                # Handle critical errors that should stop processing
                if response.status_code == 402:
                    progress.update(task, last_result="[red]Insufficient funds - stopping[/red]")
                    console.print("\n[red]❌ Critical Error: Insufficient funds (402)[/red]")
                    console.print("[yellow]⚠️  Processing stopped. Please check your OpenRouter account balance.[/yellow]")
                    break
                
                elif response.status_code == 401:
                    progress.update(task, last_result="[red]Invalid API key - stopping[/red]")
                    console.print("\n[red]❌ Critical Error: Invalid API key (401)[/red]")
                    console.print("[yellow]⚠️  Processing stopped. Please check your OPENROUTER_API_KEY.[/yellow]")
                    break
                
                elif response.status_code >= 500:
                    progress.update(task, last_result="[red]Server error - stopping[/red]")
                    console.print(f"\n[red]❌ Critical Error: Server error ({response.status_code})[/red]")
                    console.print("[yellow]⚠️  OpenRouter server issues. Processing stopped.[/yellow]")
                    break
                
                elif response.status_code >= 400:
                    # Log API errors for debugging
                    console.print(f"\n[red]⚠️ API Error {response.status_code} for {imagepath}[/red]")
                    try:
                        error_content = response.json()
                        console.print(f"[dim]Error details: {error_content.get('error', {}).get('message', 'Unknown error')}[/dim]")
                        console.print(f"[dim]Full error: {str(error_content)[:500]}[/dim]")
                    except Exception:
                        console.print(f"[dim]HTTP {response.status_code}: {response.text[:500]}[/dim]")
                        if len(response.text) > 500:
                            console.print(f"[dim]... (truncated, full length: {len(response.text)} chars)[/dim]")
                    
                    # Add empty entry to preserve zero results in table
                    # Try to get usage even for errors (some errors still have usage data)
                    try:
                        error_usage = response.json().get("usage", {})
                    except Exception:
                        error_usage = {}
                    
                    results[imagepath].append({
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
                    
                    progress.update(task, 
                                   advance=1,
                                   last_result=f"[red]API Error {response.status_code}[/red]")
                    continue
                
                response_json = response.json()
                
                # Track actual model provider used by OpenRouter
                if "provider" in response_json:
                    actual_provider = response_json["provider"]
                    config.additional_config["actual_model_providers"].add(actual_provider)
                
                if "choices" not in response_json or not response_json["choices"]:
                    # Log missing response for debugging
                    console.print(f"\n[red]⚠️ No response choices for {imagepath}[/red]")
                    console.print(f"[dim]Response: {str(response_json)[:500]}[/dim]")
                    if len(str(response_json)) > 500:
                        console.print(f"[dim]... (truncated, full length: {len(str(response_json))} chars)[/dim]")
                    
                    # Add empty entry to preserve zero results in table
                    usage = response_json.get("usage", {})
                    results[imagepath].append({
                        "student_full_name": "",
                        "university_id": "",
                        "section_number": "",
                        "_no_response": True,
                        "_token_usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                    })
                    
                    progress.update(task, 
                                   advance=1,
                                   last_result="[red]No response[/red]")
                    continue
                    
                content = response_json["choices"][0]["message"]["content"]
                choice = response_json["choices"][0]
                
                # Check for empty content with no finish reason (likely cold start/warm-up)
                if not content and choice.get("finish_reason") is None:
                    console.print(f"\n[yellow]⚠️ Empty response for {imagepath} (likely cold start)[/yellow]")
                    console.print(f"[dim]Finish reason: {choice.get('finish_reason')}[/dim]")
                    console.print(f"[dim]Completion tokens: {response_json.get('usage', {}).get('completion_tokens', 'unknown')}[/dim]")
                    console.print(f"[dim]Prompt tokens: {response_json.get('usage', {}).get('prompt_tokens', 'unknown')}[/dim]")
                    
                    # Implement retry for empty responses (cold start/warm-up issue)
                    console.print("[dim]Retrying in 10 seconds (cold start recovery)...[/dim]")
                    progress.update(task, last_result="[yellow]Retrying cold start...[/yellow]")
                    time.sleep(10)
                    
                    # Retry the request
                    retry_response = create_completion(model_name, inference_config, imagepath)
                    
                    if retry_response.status_code == 200:
                        retry_response_json = retry_response.json()
                        
                        # Track provider from retry response too
                        if "provider" in retry_response_json:
                            retry_provider = retry_response_json["provider"]
                            config.additional_config["actual_model_providers"].add(retry_provider)
                        
                        if "choices" in retry_response_json and retry_response_json["choices"]:
                            retry_content = retry_response_json["choices"][0]["message"]["content"]
                            if retry_content:  # Success on retry
                                console.print("[green]✓ Retry successful![/green]")
                                content = retry_content
                                response_json = retry_response_json
                                choice = retry_response_json["choices"][0]
                            else:
                                console.print("[red]Retry also returned empty content[/red]")
                                # Add empty entry after failed retry
                                results[imagepath].append({
                                    "student_full_name": "",
                                    "university_id": "",
                                    "section_number": "",
                                    "_empty_response": True,
                                    "_retry_failed": True
                                })
                                progress.update(task, advance=1, last_result="[red]Empty after retry[/red]")
                                continue
                        else:
                            console.print("[red]Retry returned malformed response[/red]")
                            results[imagepath].append({
                                "student_full_name": "",
                                "university_id": "",
                                "section_number": "",
                                "_empty_response": True,
                                "_retry_failed": True
                            })
                            progress.update(task, advance=1, last_result="[red]Retry failed[/red]")
                            continue
                    else:
                        console.print(f"[red]Retry failed with status {retry_response.status_code}[/red]")
                        results[imagepath].append({
                            "student_full_name": "",
                            "university_id": "",
                            "section_number": "",
                            "_empty_response": True,
                            "_retry_failed": True
                        })
                        progress.update(task, advance=1, last_result="[red]Retry failed[/red]")
                        continue
                
                # Parse response based on model format
                if isinstance(response_format, str):
                    json_obj = parse_response_content(content, response_format)
                else:
                    json_obj = None
                
                if json_obj is None:
                    # Log the parsing failure for debugging
                    console.print(f"\n[red]⚠️ Parse failed for {imagepath}[/red]")
                    console.print(f"[dim]Content type: {type(content)}[/dim]")
                    console.print(f"[dim]Content length: {len(content) if content else 'None'}[/dim]")
                    if content:
                        console.print(f"[dim]Raw response: '{content[:1000]}'[/dim]")
                        if len(content) > 1000:
                            console.print(f"[dim]... (truncated, full length: {len(content)} chars)[/dim]")
                    else:
                        console.print("[dim]Content is empty or None[/dim]")
                        console.print(f"[dim]Full response: {str(response_json)[:800]}[/dim]")
                    
                    # Add empty entry to preserve zero results in table
                    usage = response_json.get("usage", {})
                    results[imagepath].append({
                        "student_full_name": "",
                        "university_id": "",
                        "section_number": "",
                        "_parse_failed": True,
                        "_token_usage": {
                            "prompt_tokens": usage.get("prompt_tokens", 0),
                            "completion_tokens": usage.get("completion_tokens", 0),
                            "total_tokens": usage.get("total_tokens", 0)
                        }
                    })
                    
                    progress.update(task, 
                                   advance=1,
                                   last_result="[red]Parse failed[/red]")
                    continue
                
                # Standardize field names for consistency
                if "ufid" in json_obj:
                    json_obj["university_id"] = json_obj.pop("ufid", "")
                
                # Store token usage data for cost calculation
                usage = response_json.get("usage", {})
                generation_id = response_json.get("id")
                
                # Store token usage data (we'll update with precise costs later)
                json_obj["_token_usage"] = {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                    "generation_id": generation_id  # Store for batch processing later
                }
                
                results[imagepath].append(json_obj)
                successful_images += 1
                
                # Save results incrementally for streaming updates
                results_dict = dict(results)
                manager.save_results(config.run_name, results_dict)
                
                # Create concise success message
                name = json_obj.get('student_full_name', 'N/A')
                ufid = json_obj.get('university_id', '')
                if ufid:
                    result_msg = f"✓ {name} (ID: {ufid})"
                else:
                    result_msg = f"✓ {name}"
                
                # Update progress with success rate and latest result
                success_rate = f"{successful_images/i*100:.1f}%"
                progress.update(task, 
                               advance=1, 
                               success_rate=success_rate,
                               last_result=result_msg[:40] + "..." if len(result_msg) > 40 else result_msg)
                    
            except Exception as e:
                # Log the exception for debugging
                console.print(f"\n[red]⚠️ Exception processing {imagepath}[/red]")
                console.print(f"[dim]Error: {str(e)}[/dim]")
                
                # Add empty entry to preserve zero results in table
                results[imagepath].append({
                    "student_full_name": "",
                    "university_id": "",
                    "section_number": "",
                    "_exception": str(e)
                })
                
                progress.update(task, 
                               advance=1,
                               last_result=f"[red]Error: {str(e)[:20]}...[/red]")
                continue
    
    # Calculate actual runtime
    end_time = time.time()
    runtime_seconds = end_time - start_time
    runtime_formatted = format_runtime(runtime_seconds)
    
    # Update config with actual runtime
    # Create config dict and add extracted model information
    config_dict = config.to_dict()
    config_dict["environment"]["runtime"] = runtime_formatted
    config_dict["additional"]["actual_runtime_seconds"] = runtime_seconds
    
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