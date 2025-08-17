#!/usr/bin/env python3
"""
OpenRouter API interactions and data processing utilities.
"""
import os
import json
import time
import requests
from typing import Dict, Any, Optional, List
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from ..core.prompts import STUDENT_EXTRACTION_PROMPT

console = Console()

# Prompt is now centralized in core.prompts module

# Model-specific configurations for special cases
MODEL_CONFIG_OVERRIDES = {
    "z-ai/glm-4.5v": {
        "response_format": "box",  # Uses <|begin_of_box|> format
    },
    # Add more overrides as needed for models with special requirements
}


def create_completion(model_name: str, config: Dict[str, Any], imagepath: str):
    """Create a completion request to OpenRouter."""
    from .cli import filepath_to_base64
    
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
                            "text": STUDENT_EXTRACTION_PROMPT
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
            **config,
        }
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
        
        # Try to extract any JSON-like structure
        json_match = re.search(r'(\{[^{}]*"student_full_name"[^{}]*\})', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
                
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


def filter_vision_models(models: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter models to only include vision-capable models."""
    vision_models = []
    for model in models:
        # Check if model supports images in description or architecture
        description = model.get("description", "").lower()
        architecture = model.get("architecture", {})
        modality = architecture.get("modality", "") if architecture else ""
        
        # Look for vision indicators
        vision_indicators = ["vision", "image", "multimodal", "vlm", "visual"]
        if any(indicator in description for indicator in vision_indicators) or \
           any(indicator in modality.lower() for indicator in vision_indicators):
            vision_models.append(model)
    
    return vision_models


def batch_update_generation_costs(results: Dict) -> Dict:
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
    
    updated_count = 0
    
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
        
        task = progress.add_task("Updating costs...", total=len(generation_ids_to_process))
        
        for filepath, generation_id in generation_ids_to_process:
            try:
                # Query generation API for precise stats
                stats_response = requests.get(
                    f"https://openrouter.ai/api/v1/generation?id={generation_id}",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=5
                )
                
                # Always remove generation_id first
                result = results[filepath][0]
                token_usage = result["_token_usage"]
                token_usage.pop("generation_id", None)
                
                if stats_response.status_code == 200:
                    stats_data = stats_response.json()
                    data = stats_data.get("data", {})
                    
                    if data:  # Make sure we got actual data
                        # Update with native token counts and actual cost
                        if "native_tokens_prompt" in data:
                            token_usage["prompt_tokens"] = data["native_tokens_prompt"]
                        if "native_tokens_completion" in data:
                            token_usage["completion_tokens"] = data["native_tokens_completion"]
                        
                        actual_cost = data.get("total_cost", 0.0)
                        if actual_cost > 0:
                            token_usage["actual_cost"] = actual_cost
                            updated_count += 1
                            progress.update(task, advance=1, description=f"Updated {updated_count} costs")
                        else:
                            progress.update(task, advance=1, description=f"Updated {updated_count} costs (no cost data)")
                    else:
                        progress.update(task, advance=1, description=f"Updated {updated_count} costs (empty response)")
                
                else:
                    # Log the failure for debugging
                    if stats_response.status_code == 404:
                        progress.update(task, advance=1, description=f"Updated {updated_count} costs (404 not found)")
                    else:
                        progress.update(task, advance=1, description=f"Updated {updated_count} costs (API error {stats_response.status_code})")
            
            except Exception as e:
                # Remove generation_id and log error
                results[filepath][0]["_token_usage"].pop("generation_id", None)
                progress.update(task, advance=1, description=f"Updated {updated_count} costs (exception)")
                console.print(f"[dim]Error for {generation_id}: {str(e)}[/dim]")
    
    if updated_count > 0:
        console.print(f"[green]✅ Updated {updated_count}/{len(generation_ids_to_process)} results with precise costs[/green]")
    else:
        console.print("[yellow]⚠️ Could not get precise costs for any results[/yellow]")
    
    return results