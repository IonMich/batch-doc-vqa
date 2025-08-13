#!/usr/bin/env python3
"""
General OpenRouter inference script for vision models.
Supports any vision model available on OpenRouter with configurable parameters.
"""
import os
import base64
import re
import json
import time
import dotenv
import requests
import argparse
from collections import defaultdict
from typing import Dict, Any, Optional, List

from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn
from rich import print as rprint

from run_manager import RunManager, RunConfig

dotenv.load_dotenv()

console = Console()

# Prompt template for student information extraction
STUDENT_EXTRACTION_PROMPT = """Extract the student information from this quiz submission. Return ONLY valid JSON in this format:
{
    "student_full_name": "Full name of the student",
    "ufid": "8-digit UFID number if present, empty string if missing",
    "section_number": "5-digit section number"
}

Example:
{
    "student_full_name": "John Doe",
    "ufid": "12345678",
    "section_number": "11900"
}

If UFID is not visible in the image, use an empty string for ufid."""

# Model-specific configurations for special cases
MODEL_CONFIG_OVERRIDES = {
    "z-ai/glm-4.5v": {
        "response_format": "box",  # Uses <|begin_of_box|> format
        "max_tokens": 2048,
    },
    # Add more overrides as needed for models with special requirements
}


def filepath_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_imagepaths(folder, pattern):
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if re.match(pattern, file):
                images.append(os.path.join(root, file))
    # sort by integers in the filename
    images.sort(key=natural_sort_key)
    return images


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def create_completion(model_name: str, config: Dict[str, Any], imagepath: str):
    """Create a completion request to OpenRouter."""
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


def interactive_provider_model_selection(show_all_providers: bool = False):
    """Interactive provider and model selection flow."""
    console.print("\n[bold blue]🔍 Fetching OpenRouter Providers and Models...[/bold blue]")
    
    # Fetch data
    providers = fetch_openrouter_providers()
    models = fetch_openrouter_models()
    
    if not providers or not models:
        console.print("[red]Failed to fetch data from OpenRouter API[/red]")
        return None
    
    # Filter to vision models only
    vision_models = filter_vision_models(models)
    
    # Group models by provider
    models_by_provider = {}
    for model in vision_models:
        model_id = model.get("id", "")
        if "/" in model_id:
            provider_slug = model_id.split("/")[0]
        else:
            provider_slug = "unknown"
        
        if provider_slug not in models_by_provider:
            models_by_provider[provider_slug] = []
        models_by_provider[provider_slug].append(model)
    
    # Filter providers to only show those with vision models
    providers_with_vision = []
    for provider in providers:
        provider_slug = provider.get("slug", "")
        if provider_slug in models_by_provider:
            providers_with_vision.append(provider)
    
    # Handle special cases: model namespaces that don't have corresponding provider entities
    # but are hosted by safe providers (e.g., qwen models hosted by alibaba)
    special_namespaces = {
        "qwen": {"name": "Qwen (Alibaba)", "may_train_on_data": False},  # Qwen is Alibaba's LLM
        "mistralai": {"name": "Mistral AI", "may_train_on_data": False},
        "meta-llama": {"name": "Meta Llama", "may_train_on_data": False}
    }
    
    for namespace, info in special_namespaces.items():
        if namespace in models_by_provider:
            # Create a virtual provider entry
            virtual_provider = {
                "slug": namespace,
                "name": info["name"],
                "may_train_on_data": info["may_train_on_data"]
            }
            providers_with_vision.append(virtual_provider)
    
    # Filter out providers that may train on data
    providers_no_training = []
    for provider in providers_with_vision:
        may_train = provider.get("may_train_on_data", True)  # Default to True for safety
        if not may_train:  # Only include providers that don't train on data
            providers_no_training.append(provider)
    
    console.print(f"\n[dim]Filtered out {len(providers_with_vision) - len(providers_no_training)} providers that may train on your data[/dim]")
    providers_with_vision = providers_no_training
    
    # Default filter: show only popular providers unless show_all is True
    # Based on analysis: these providers exist, don't train on data, and have vision models
    popular_provider_slugs = {"openai", "anthropic", "z-ai", "qwen", "mistralai", "meta-llama", "moonshotai"}
    
    if not show_all_providers:
        filtered_providers = []
        for provider in providers_with_vision:
            provider_slug = provider.get("slug", "")
            if provider_slug in popular_provider_slugs:
                filtered_providers.append(provider)
        display_providers = filtered_providers
    else:
        display_providers = providers_with_vision
    
    # Step 1: Provider Selection
    console.print(f"\nFound [bold green]{len(vision_models)}[/bold green] vision models from [bold green]{len(providers_with_vision)}[/bold green] providers")
    
    if not show_all_providers:
        console.print(f"[dim]Showing [bold green]{len(display_providers)}[/bold green] some popular providers[/dim]")
    else:
        console.print(f"[dim]Showing all [bold green]{len(display_providers)}[/bold green] providers (safe providers only - no training on data)[/dim]")
    
    console.print("\n[bold cyan]📋 Select a Provider:[/bold cyan]")
    
    provider_table = Table(show_header=True, header_style="bold cyan")
    provider_table.add_column("#", style="cyan", width=3)
    provider_table.add_column("Provider", style="yellow")
    provider_table.add_column("Vision Models", style="green", justify="center")
    provider_table.add_column("Status", style="white")
    
    for i, provider in enumerate(display_providers, 1):
        provider_slug = provider.get("slug", "")
        provider_name = provider.get("name", provider_slug)
        model_count = len(models_by_provider.get(provider_slug, []))
        
        # Simple status indicator
        status = "✅ Available"
        
        provider_table.add_row(
            str(i),
            provider_name,
            str(model_count),
            status
        )
    
    console.print(provider_table)
    
    # Add show all option
    if not show_all_providers:
        console.print(f"\n[dim]💡 To see all {len(providers_with_vision)} providers, type 'all'[/dim]")
    
    # Get provider selection
    try:
        valid_choices = [str(i) for i in range(1, len(display_providers) + 1)] + ["q"]
        if not show_all_providers:
            valid_choices.append("all")
        
        choice = Prompt.ask(
            "\nSelect provider number, 'all' to show all providers, or 'q' to quit" if not show_all_providers else "\nSelect provider number (or 'q' to quit)",
            choices=valid_choices,
            default="q"
        )
        
        if choice == "q":
            console.print("👋 Goodbye!")
            return None
            
        if choice == "all":
            console.print("Showing all providers...\n")
            return interactive_provider_model_selection(show_all_providers=True)
            
        selected_provider = display_providers[int(choice) - 1]
        provider_slug = selected_provider.get("slug", "")
        provider_name = selected_provider.get("name", provider_slug)
        
    except (ValueError, IndexError):
        console.print("[red]Invalid selection[/red]")
        return None
    
    # Step 2: Model Selection
    provider_models = models_by_provider.get(provider_slug, [])
    console.print(f"\n[bold cyan]🤖 {provider_name} Vision Models:[/bold cyan]")
    
    model_table = Table(show_header=True, header_style="bold cyan")
    model_table.add_column("#", style="cyan", width=3)
    model_table.add_column("Model ID", style="green")
    model_table.add_column("Input $/M", style="yellow", justify="right")
    model_table.add_column("Output $/M", style="orange3", justify="right")
    model_table.add_column("Image $/K", style="magenta", justify="right")
    model_table.add_column("Context", style="blue", justify="right")
    
    for i, model in enumerate(provider_models, 1):
        model_id = model.get("id", "")
        context_length = model.get("context_length", "Unknown")
        pricing = model.get("pricing", {})
        
        # Extract pricing info and convert to per-million tokens
        input_cost = pricing.get("prompt", "0")
        output_cost = pricing.get("completion", "0") 
        image_cost = pricing.get("image", "0")
        
        # Convert to per-million format (prices are per token, multiply by 1M)
        try:
            input_per_m = f"${float(input_cost) * 1_000_000:.2f}" if input_cost != "0" else "Free"
        except (ValueError, TypeError):
            input_per_m = "Unknown"
            
        try:
            output_per_m = f"${float(output_cost) * 1_000_000:.2f}" if output_cost != "0" else "Free"
        except (ValueError, TypeError):
            output_per_m = "Unknown"
            
        try:
            image_per_k = f"${float(image_cost) * 1_000:.2f}" if image_cost != "0" else "-"
        except (ValueError, TypeError):
            image_per_k = "-"
        
        model_table.add_row(
            str(i),
            model_id,
            input_per_m,
            output_per_m,
            image_per_k,
            str(context_length)
        )
    
    console.print(model_table)
    
    # Get model selection
    try:
        model_choice = Prompt.ask(
            f"\nSelect model number (1-{len(provider_models)}), 'b' to go back, or 'q' to quit",
            choices=[str(i) for i in range(1, len(provider_models) + 1)] + ["b", "q"],
            default="b"
        )
        
        if model_choice == "b":
            console.print("Going back to provider selection...\n")
            return interactive_provider_model_selection(show_all_providers)  # Pass the current filter state
            
        if model_choice == "q":
            console.print("👋 Goodbye!")
            return None
            
        selected_model = provider_models[int(model_choice) - 1]
        model_id = selected_model.get("id", "")
        
        console.print(f"\n✅ Selected: [bold green]{model_id}[/bold green]")
        
        # Show usage example
        console.print("\n[bold blue]💡 To run this model:[/bold blue]")
        console.print(f"uv run python openrouter_inference.py --model [green]{model_id}[/green]")
        console.print(f"uv run python openrouter_inference.py --model [green]{model_id}[/green] [blue]--interactive[/blue]")
        
        return model_id
        
    except (ValueError, IndexError):
        console.print("[red]Invalid selection[/red]")
        return None


def interactive_config_prompt(model_name: str) -> Dict[str, Any]:
    """Interactive prompt for model configuration when not provided via CLI."""
    
    console.print("\n[bold blue]🔧 Model Configuration[/bold blue]")
    console.print(f"Setting up configuration for: [bold green]{model_name}[/bold green]\n")
    
    # Parse org from model name
    org = model_name.split("/")[0] if "/" in model_name else "unknown"
    
    # Model size
    size_options = ["Unknown", "500M", "1B", "2B", "7B", "8B", "11B", "72B", "90B", "Custom"]
    size_table = Table(show_header=False, box=None)
    for i, size in enumerate(size_options, 1):
        size_table.add_row(f"[cyan]{i}[/cyan]", size)
    
    console.print("[bold]Model Size:[/bold]")
    console.print(size_table)
    size_choice = Prompt.ask(
        "Choose model size", 
        choices=[str(i) for i in range(1, len(size_options) + 1)],
        default="1"
    )
    model_size = size_options[int(size_choice) - 1]
    
    if model_size == "Custom":
        model_size = Prompt.ask("Enter custom model size")
    
    # Open weights
    open_weights = Confirm.ask("Is this an open-weights model?", default=False)
    
    # License info
    if open_weights:
        license_options = ["Apache 2.0", "MIT", "GPL v3", "Custom/Other"]
        license_table = Table(show_header=False, box=None)
        for i, license_type in enumerate(license_options, 1):
            license_table.add_row(f"[cyan]{i}[/cyan]", license_type)
        
        console.print("\n[bold]License Type:[/bold]")
        console.print(license_table)
        license_choice = Prompt.ask(
            "Choose license type",
            choices=[str(i) for i in range(1, len(license_options) + 1)],
            default="1"
        )
        license_info = license_options[int(license_choice) - 1]
        
        if license_info == "Custom/Other":
            license_info = Prompt.ask("Enter license information")
    else:
        license_info = "Proprietary"
    
    return {
        "model_size": model_size,
        "open_weights": open_weights, 
        "license_info": license_info,
    }


def format_runtime(seconds: float) -> str:
    """Format runtime in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"


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
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("[bold green]{task.fields[success_rate]}"),
        TextColumn("•"),
        TextColumn("{task.fields[last_result]}", style="dim")
    ) as progress:
        
        task = progress.add_task(
            "Processing images...", 
            total=total_images,
            success_rate="0%",
            last_result=""
        )
        
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
                    except:
                        console.print(f"[dim]HTTP {response.status_code}: {response.text[:500]}[/dim]")
                        if len(response.text) > 500:
                            console.print(f"[dim]... (truncated, full length: {len(response.text)} chars)[/dim]")
                    
                    # Add empty entry to preserve zero results in table
                    # Try to get usage even for errors (some errors still have usage data)
                    try:
                        error_usage = response.json().get("usage", {})
                    except:
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
    import yaml
    from pathlib import Path
    config_path = Path(run_dir) / "config.yaml"
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    # Final results save (incremental saves were done during processing)
    results_dict = dict(results)
    manager.save_results(config.run_name, results_dict)
    
    # Display results with rich formatting
    console.print("\n[bold green]✅ Inference Complete![/bold green]")
    
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
    for filepath, result_list in updated_results.items():
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
    
    console = Console()
    
    # Wait a bit for all generation stats to be ready
    console.print(f"[dim]Found {len(generation_ids_to_process)} generation IDs to process[/dim]")
    console.print(f"[dim]Waiting 3 seconds for generation stats to be ready...[/dim]")
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


def list_model_overrides():
    """List models with special configuration overrides."""
    if not MODEL_CONFIG_OVERRIDES:
        print("No model-specific overrides configured.")
        return
        
    print("Models with Special Configuration Overrides:")
    print("=" * 50)
    for model_name, overrides in MODEL_CONFIG_OVERRIDES.items():
        print(model_name)
        for key, value in overrides.items():
            print(f"  {key}: {value}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using OpenRouter vision models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run GLM-4.5V
  python openrouter_inference.py --model z-ai/glm-4.5v
  
  # List models with special configurations
  python openrouter_inference.py --list-overrides
        """
    )
    
    parser.add_argument("--model", type=str, help="OpenRouter model name (e.g., 'z-ai/glm-4.5v')")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0.0 for deterministic output)")
    parser.add_argument("--max-tokens", type=int, default=4096, help="Max tokens (default: 4096)")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p sampling (default: 1.0 - OpenRouter default)")
    parser.add_argument("--model-size", type=str, help="Model size (e.g., '7B', '72B')")
    parser.add_argument("--open-weights", action="store_true", help="Mark as open-weights model")
    parser.add_argument("--license-info", type=str, help="License information")
    parser.add_argument("--interactive", "-i", action="store_true", help="Use interactive prompts for missing configuration")
    parser.add_argument("--list-overrides", action="store_true", help="List models with special configurations and exit")
    
    args = parser.parse_args()
    
    if args.list_overrides:
        list_model_overrides()
        return
    
    if not args.model:
        # Interactive provider and model selection
        selected_model = interactive_provider_model_selection()
        if not selected_model:
            return
        # Set the selected model for inference
        args.model = selected_model
    
    # Check for API key
    if not os.getenv('OPENROUTER_API_KEY'):
        print("Error: OPENROUTER_API_KEY environment variable not set")
        return
    
    run_name = run_openrouter_inference(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        model_size=args.model_size,
        open_weights=args.open_weights,
        license_info=args.license_info,
        interactive=args.interactive
    )
    
    print(f"\nRun completed: {run_name}")
    print("Generate benchmark table with:")
    print(f"uv run python generate_benchmark_table.py --patterns {args.model.split('/')[0]}")


if __name__ == "__main__":
    main()