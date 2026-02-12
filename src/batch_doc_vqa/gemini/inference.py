#!/usr/bin/env python3
"""
Gemini inference engine and orchestration.
"""
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console

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
from ..core.prompts import STUDENT_EXTRACTION_PROMPT

console = Console()


def remove_thinking_content(content: str) -> str:
    """Remove thinking tags and content - handle various thinking formats (like OpenRouter)."""
    import re
    thinking_patterns = [
        r'◁think▷.*?◁/think▷',  # ◁think▷...◁/think▷
        r'<think>.*?</think>',   # <think>...</think>  
        r'<thinking>.*?</thinking>',  # <thinking>...</thinking>
        r'◁think▷.*?(?=\{|So my JSON|JSON response:|Response:)',  # ◁think▷ until JSON starts
    ]
    
    for pattern in thinking_patterns:
        content = re.sub(pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    return content.strip()


def run_gemini_inference(model_id: str = "gemini-2.5-flash",
                        temperature: float = 0.0,
                        model_size: Optional[str] = None,
                        open_weights: Optional[bool] = None,
                        license_info: Optional[str] = None,
                        interactive: bool = False,
                        pricing: Optional[dict] = None,
                        images_dir: str = "imgs/q11",
                        doc_info_file: Optional[str] = None,
                        pages: Optional[list[int]] = None):
    """Run inference using Google Gemini API."""
    
    # Start timing
    start_time = time.time()
    
    # Parse model name for Gemini
    # Examples: "gemini-2.5-flash" → org="google", model="gemini-2.5-flash"
    org = "google"
    model = model_id
    
    # Always prompt for pricing if not provided (essential for cost tracking)
    if pricing is None:
        from .ui import get_model_pricing
        pricing = get_model_pricing(model_id)
    
    # Interactive configuration if missing other info and interactive mode
    if interactive and (model_size is None or open_weights is None or license_info is None):
        from .ui import interactive_config_prompt
        interactive_config = interactive_config_prompt(model_id)
        model_size = model_size or interactive_config["model_size"]
        open_weights = open_weights if open_weights is not None else interactive_config["open_weights"]
        license_info = license_info or interactive_config["license_info"]
    
    # Set defaults for Gemini models
    if model_size is None:
        model_size = "Unknown"  # Gemini doesn't publish model sizes
    open_weights = open_weights if open_weights is not None else False  # Gemini is proprietary
    license_info = license_info or "Proprietary (Google)"
    
    selected_pages = pages if pages else [1, 3]

    # Create run configuration
    additional_config = {
        "base_url": "https://generativelanguage.googleapis.com/",
        "endpoint_type": "generateContent",
        "model_id": model_id,
        "concurrency": 1,
        "prompt_template": STUDENT_EXTRACTION_PROMPT,
        "thinking_config": "thinking" in model_id.lower(),
        "images_dir": images_dir,
        "doc_info_file": doc_info_file,
        "pages": selected_pages,
    }
    
    # Add pricing to config if available (like OpenRouter)
    if pricing:
        additional_config["model_pricing"] = {
            "prompt": str(pricing["input_per_million"] / 1_000_000),  # Convert to per-token like OpenRouter
            "completion": str(pricing["output_per_million"] / 1_000_000),
            "input_per_million": pricing["input_per_million"],
            "output_per_million": pricing["output_per_million"],
        }
    
    config = RunConfig(
        org=org,
        model=model,
        model_size=model_size,
        open_weights=open_weights,
        license_info=license_info,
        api_provider="Google Gemini API",
        use_structured_output=True,
        use_regex_patterns=False,
        temperature=temperature,
        max_tokens=None,  # Gemini handles this automatically
        runtime_environment="TBD",  # Will be updated with actual runtime
        additional_config=additional_config
    )

    for warning_line in build_git_dirty_warning_lines(config):
        console.print(warning_line)
    
    # Create run directory
    manager = RunManager()
    run_dir = manager.create_run_directory(config)
    
    print(f"Starting Gemini inference run: {config.run_name}")
    print(f"Model: {model_id}")
    print(f"Run directory: {run_dir}")
    
    # Setup inference parameters - use same defaults as OpenRouter
    if doc_info_file:
        imagepaths = get_imagepaths_from_doc_info(
            doc_info_file,
            images_dir=images_dir,
            pages=selected_pages,
        )
    else:
        pattern = r"doc-\d+-page-[" + "".join([str(p) for p in selected_pages]) + "]-[A-Z0-9]+.png"
        imagepaths = get_imagepaths(images_dir, pattern)

    # Run inference with rich progress tracking
    results = defaultdict(list)
    total_images = len(imagepaths)
    if total_images == 0:
        console.print("[red]❌ No images matched the selected dataset/pages.[/red]")
        return ""
    successful_images = 0
    
    # Create progress bar with live status (matching OpenRouter)
    with create_inference_progress() as progress:
        task = add_inference_task(progress, total_images)
        
        # Process one image at a time to keep progress + per-image metadata.
        for i, imagepath in enumerate(imagepaths, 1):
            image_started_epoch = time.time()
            image_started_at_utc = datetime.now(timezone.utc).isoformat()
            try:
                # Process single image with pricing information
                result = process_single_image(model_id, temperature, imagepath, pricing)
                
                if result:
                    image_finished_at_utc = datetime.now(timezone.utc).isoformat()
                    elapsed_seconds = max(0.0, time.time() - image_started_epoch)
                    result["_timing"] = {
                        "started_at_utc": image_started_at_utc,
                        "finished_at_utc": image_finished_at_utc,
                        "elapsed_seconds": elapsed_seconds,
                    }
                    results[imagepath].append(result)
                    successful_images += 1
                    
                    # Save results incrementally for streaming updates
                    results_dict = dict(results)
                    manager.save_results(config.run_name, results_dict)
                    
                    # Create concise success message
                    name = result.get('student_full_name', 'N/A')
                    ufid = result.get('university_id', '')
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
                else:
                    # Handle failed processing
                    progress.update(task, advance=1, last_result="[red]Failed[/red]")
                    
            except Exception as e:
                # Log the exception for debugging
                console.print(f"\n[red]⚠️ Exception processing {imagepath}[/red]")
                console.print(f"[dim]Error: {str(e)}[/dim]")
                
                # Add empty entry to preserve zero results
                image_finished_at_utc = datetime.now(timezone.utc).isoformat()
                elapsed_seconds = max(0.0, time.time() - image_started_epoch)
                results[imagepath].append({
                    "student_full_name": "",
                    "university_id": "",
                    "section_number": "",
                    "_exception": str(e),
                    "_timing": {
                        "started_at_utc": image_started_at_utc,
                        "finished_at_utc": image_finished_at_utc,
                        "elapsed_seconds": elapsed_seconds,
                    },
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
    config_dict = config.to_dict()
    config_dict["environment"]["runtime"] = runtime_formatted
    config_dict["additional"]["actual_runtime_seconds"] = runtime_seconds
    
    # Save updated config and refresh manifest metadata
    manager.save_run_config(config.run_name, config_dict)
    
    # Final results save
    results_dict = dict(results)
    manager.save_results(config.run_name, results_dict)
    
    # Display results with rich formatting (matching OpenRouter)
    console.print("\n[bold green]✅ Inference Complete![/bold green]")
    
    from rich.table import Table
    results_table = Table(show_header=False, box=None)
    results_table.add_row("[cyan]Processed:[/cyan]", f"{successful_images}/{total_images} images")
    results_table.add_row("[cyan]Success rate:[/cyan]", f"{successful_images/total_images*100:.1f}%")
    results_table.add_row("[cyan]Runtime:[/cyan]", f"[bold]{runtime_formatted}[/bold]")
    results_table.add_row("[cyan]API Provider:[/cyan]", "Google Gemini")
    results_table.add_row("[cyan]Results saved:[/cyan]", f"{run_dir}/results.json")
    results_table.add_row("[cyan]Run name:[/cyan]", f"[bold]{config.run_name}[/bold]")
    
    console.print(results_table)
    
    return config.run_name


def process_single_image(model_id: str, temperature: float, imagepath: str, pricing: dict = None):
    """Process a single image with Gemini API."""
    import os
    import json
    from google import genai
    from google.genai import types
    from PIL import Image
    from .api import parse_json, prompt
    
    # Setup client
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")
    
    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    
    # Setup config
    config = types.GenerateContentConfig(
        temperature=temperature,
        thinking_config=types.ThinkingConfig(
            include_thoughts=True,  # Let model think and include thoughts, we'll filter them out
        ) if "thinking" in model_id.lower() else None,
    )
    
    try:
        # Create completion using base64 format (like OpenRouter for consistency)
        from ..core import filepath_to_base64
        base64_image = filepath_to_base64(imagepath)
        
        image_content = {
            "inlineData": {
                "mimeType": "image/png",
                "data": base64_image
            }
        }
        
        response = client.models.generate_content(
            model=model_id,
            contents=[prompt, image_content],  # Text first, image second (like OpenRouter)
            config=config,
        )
        
        # Extract token usage data
        usage = getattr(response, "usage_metadata", None)
        token_usage = {}
        actual_cost = None
        
        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0)
            candidates_tokens = getattr(usage, "candidates_token_count", 0)
            total_tokens = getattr(usage, "total_token_count", 0)
            
            token_usage = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": candidates_tokens,
                "total_tokens": total_tokens,
                "cached_content_token_count": getattr(usage, "cached_content_token_count", 0),
                "thoughts_token_count": getattr(usage, "thoughts_token_count", 0),
            }
            
            # Calculate cost if pricing is available
            if pricing and prompt_tokens and candidates_tokens:
                input_cost = (prompt_tokens / 1_000_000) * pricing.get("input_per_million", 0)
                output_cost = (candidates_tokens / 1_000_000) * pricing.get("output_per_million", 0)
                actual_cost = input_cost + output_cost
                token_usage["actual_cost"] = actual_cost
                token_usage["input_cost"] = input_cost
                token_usage["output_cost"] = output_cost
        
        # Parse response
        response_parts = (
            response.model_dump(mode="json")
            .get("candidates")[0]
            .get("content")
            .get("parts")
        )
        model_responses = [r for r in response_parts if not r.get("thought")]
        json_str = model_responses[0].get("text")
        
        # Apply JSON parsing to clean up markdown formatting and thinking content
        json_str = parse_json(json_str)
        
        # Remove thinking content like OpenRouter does (for consistency)
        json_str = remove_thinking_content(json_str)
        json_obj = json.loads(json_str)
        
        # Replace ufid with university_id for consistency
        if "ufid" in json_obj:
            json_obj["university_id"] = json_obj.pop("ufid")
        
        # Add token usage data
        if token_usage:
            json_obj["_token_usage"] = token_usage
            
        return json_obj
        
    except Exception as e:
        console.print(f"[red]Error processing {imagepath}: {e}[/red]")
        return None


if __name__ == "__main__":
    run_gemini_inference()
