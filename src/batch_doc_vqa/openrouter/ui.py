#!/usr/bin/env python3
"""
OpenRouter interactive user interface components.
"""
from datetime import datetime
import requests
from typing import Dict, Any, List, Optional
from collections import defaultdict
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

from .api import fetch_openrouter_providers, fetch_openrouter_models, filter_vision_models
from ..core.run_manager import RunManager

console = Console()


def extract_organization(model_id: str) -> str:
    """Extract organization from model ID (e.g., 'anthropic/claude-3' -> 'anthropic')."""
    return model_id.split("/")[0] if "/" in model_id else "unknown"


def group_models_by_organization(models: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group models by their organization (model creator)."""
    organizations = defaultdict(list)
    
    for model in models:
        model_id = model.get("id", "")
        org = extract_organization(model_id)
        organizations[org].append(model)
    
    return dict(organizations)


def _format_benchmark_date(run_info: Dict[str, Any]) -> str:
    """Format a run timestamp for compact table display."""
    timestamp_iso = run_info.get("timestamp_iso")
    if timestamp_iso:
        try:
            dt = datetime.fromisoformat(timestamp_iso.replace("Z", "+00:00"))
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            pass

    timestamp = run_info.get("timestamp")
    if timestamp:
        try:
            dt = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            return str(timestamp)

    return "Unknown"


def build_last_benchmarked_lookup() -> Dict[str, str]:
    """Build model-id -> most recent benchmark date from local run history."""
    try:
        runs = RunManager().list_runs()
    except Exception:
        return {}

    last_benchmarked: Dict[str, str] = {}
    for run in runs:
        config = run.get("config", {})
        run_info = config.get("run_info", {})
        model_cfg = config.get("model", {})
        additional_cfg = config.get("additional", {})

        model_ids: List[str] = []
        org = model_cfg.get("org")
        model_name = model_cfg.get("model")
        if org and model_name:
            model_ids.append(f"{org}/{model_name}")

        additional_model_name = additional_cfg.get("model_name")
        if isinstance(additional_model_name, str) and additional_model_name:
            model_ids.append(additional_model_name)

        if not model_ids:
            continue

        date_value = _format_benchmark_date(run_info)
        # Runs are sorted newest-first, so first hit is the latest for each model.
        for model_id in model_ids:
            if model_id not in last_benchmarked:
                last_benchmarked[model_id] = date_value

    return last_benchmarked


def fetch_model_endpoints(model_id: str) -> Optional[Dict[str, Any]]:
    """Fetch endpoint information for a specific model."""
    if "/" not in model_id:
        return None
        
    author, slug = model_id.split("/", 1)
    url = f"https://openrouter.ai/api/v1/models/{author}/{slug}/endpoints"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        console.print(f"[dim]Warning: Could not fetch endpoints for {model_id}: {e}[/dim]")
        return None


def display_provider_policies(model_id: str, providers_data: Optional[Dict[str, Any]]) -> bool:
    """Display provider data policies for a selected model. Returns True if user accepts."""
    console.print(f"\n[bold cyan]🔒 Data Policy Review for {model_id}[/bold cyan]")
    
    # Fetch endpoint data for this model
    console.print(f"[dim]Fetching provider information for {model_id}...[/dim]")
    endpoints_data = fetch_model_endpoints(model_id)
    
    if not endpoints_data or "data" not in endpoints_data:
        console.print("[yellow]⚠️  No endpoint data available - proceeding with OpenRouter's automatic provider selection[/yellow]")
        return Confirm.ask("Continue without provider policy review?", default=True)
    
    endpoints = endpoints_data["data"].get("endpoints", [])
    if not endpoints:
        console.print("[yellow]⚠️  No endpoints found - proceeding with OpenRouter's automatic provider selection[/yellow]")
        return Confirm.ask("Continue without provider policy review?", default=True)
    
    # Create providers lookup
    providers_lookup = {}
    if providers_data and "data" in providers_data:
        for provider in providers_data["data"]:
            slug = provider.get("slug", "")
            providers_lookup[slug] = provider
    
    # Show provider table with policies
    console.print(f"\n[cyan]Available providers for {model_id}:[/cyan]")
    
    policy_table = Table(show_header=True, header_style="bold cyan")
    policy_table.add_column("Provider", style="yellow", width=15)
    policy_table.add_column("Status", style="green", width=8)
    policy_table.add_column("Privacy Policy", style="blue", width=56, overflow="fold")
    policy_table.add_column("Terms of Service", style="white", width=56, overflow="fold")
    
    unique_providers = set()
    for endpoint in endpoints:
        provider_name = endpoint.get("provider_name", "Unknown")
        status = endpoint.get("status", "Unknown")
        unique_providers.add(provider_name.lower())
        
        # Try to find policy URLs
        privacy_url = "Not available"
        terms_url = "Not available"
        
        # Match provider with policy data
        for provider_slug, provider_info in providers_lookup.items():
            provider_display_name = provider_info.get("name", "").lower()
            if (provider_name.lower() in provider_display_name or 
                provider_display_name in provider_name.lower() or
                provider_slug.lower() == provider_name.lower()):
                
                privacy_url = provider_info.get("privacy_policy_url", "Not available")
                terms_url = provider_info.get("terms_of_service_url", "Not available")
                break
        
        # Show raw URLs so users can copy/paste even when terminal hyperlinks are unsupported.
        if privacy_url != "Not available":
            privacy_display = f"[link={privacy_url}]{privacy_url}[/link]"
        else:
            privacy_display = "[dim]Not available[/dim]"

        if terms_url != "Not available":
            terms_display = f"[link={terms_url}]{terms_url}[/link]"
        else:
            terms_display = "[dim]Not available[/dim]"
        
        policy_table.add_row(
            provider_name,
            str(status),
            privacy_display,
            terms_display
        )
    
    console.print(policy_table)
    
    # Show summary and get user confirmation
    provider_count = len(endpoints)
    if provider_count == 1:
        console.print(f"\n[cyan]💡 OpenRouter will use this provider for inference.[/cyan]")
        confirm_text = f"Proceed with {model_id} using this provider?"
    else:
        console.print(f"\n[cyan]💡 OpenRouter will automatically select from these {provider_count} providers during inference.[/cyan]")
        confirm_text = f"Proceed with {model_id} using these providers?"
    
    console.print("[dim]You can review the privacy policies and terms of service at the URLs above.[/dim]")
    console.print("[dim]Tip: if links are not clickable in your terminal, copy/paste the URL text directly.[/dim]")
    
    return Confirm.ask(f"\n{confirm_text}", default=True)


def interactive_organization_model_selection(show_all_orgs: bool = False):
    """Interactive organization and model selection flow."""
    console.print("\n[bold blue]🔍 Fetching OpenRouter Models...[/bold blue]")
    
    # Fetch data
    providers_data = fetch_openrouter_providers()
    models_data = fetch_openrouter_models()
    
    if not models_data:
        console.print("[red]Failed to fetch models data from OpenRouter API[/red]")
        return None
    
    all_models = models_data
    console.print(f"[green]✅ Found {len(all_models)} total models[/green]")
    
    # Filter to vision models only
    vision_models = filter_vision_models(all_models)
    console.print(f"[green]✅ Found {len(vision_models)} vision models[/green]")
    
    # Group by organization
    organizations = group_models_by_organization(vision_models)
    console.print(f"[green]✅ Found {len(organizations)} organizations with vision models[/green]")
    last_benchmarked_lookup = build_last_benchmarked_lookup()
    
    # Define organization display names and trusted organizations
    org_display_names = {
        "anthropic": "Anthropic",
        "openai": "OpenAI", 
        "google": "Google",
        "meta-llama": "Meta (Llama)",
        "qwen": "Qwen (Alibaba)",
        "mistralai": "Mistral AI",
        "microsoft": "Microsoft",
        "x-ai": "xAI (Grok)",
        "z-ai": "Zhipu AI (GLM)",
        "perplexity": "Perplexity",
        "moonshotai": "MoonshotAI",
        "amazon": "Amazon"
    }
    
    # Filter to trusted organizations by default
    trusted_orgs = list(org_display_names.keys())
    
    if not show_all_orgs:
        # Show only trusted organizations
        filtered_orgs = {k: v for k, v in organizations.items() if k in trusted_orgs}
        display_orgs = filtered_orgs
    else:
        display_orgs = organizations
    
    # Step 1: Organization Selection
    console.print(f"\nFound [bold green]{len(vision_models)}[/bold green] vision models from [bold green]{len(organizations)}[/bold green] organizations")
    
    if not show_all_orgs:
        console.print(f"[dim]Showing [bold green]{len(display_orgs)}[/bold green] trusted organizations[/dim]")
    else:
        console.print(f"[dim]Showing all [bold green]{len(display_orgs)}[/bold green] organizations[/dim]")
    
    console.print("\n[bold cyan]📋 Select an Organization:[/bold cyan]")
    
    org_table = Table(show_header=True, header_style="bold cyan")
    org_table.add_column("No.", style="cyan", width=3)
    org_table.add_column("Organization", style="yellow", width=25)
    org_table.add_column("Models", style="green", justify="center", width=10)
    org_table.add_column("Example Models", style="white", width=40)
    
    # Sort organizations by model count (descending)
    sorted_orgs = sorted(display_orgs.items(), key=lambda x: len(x[1]), reverse=True)
    
    for i, (org, org_models) in enumerate(sorted_orgs, 1):
        display_name = org_display_names.get(org, org.replace("-", " ").title())
        
        # Sort models by creation date (newest first) to show latest models
        sorted_models = sorted(org_models, key=lambda x: x.get("created", 0), reverse=True)
        
        # Extract actual model names (remove organization prefix if present)
        example_names = []
        for model in sorted_models[:3]:
            name = model.get("name", model.get("id", "Unknown"))
            # Remove organization prefix (e.g., "OpenAI: GPT-5" -> "GPT-5")
            if ":" in name and name.split(":")[0].strip().lower() in org.lower():
                clean_name = name.split(":", 1)[1].strip()
            else:
                clean_name = name
            # Truncate long names
            example_names.append(clean_name[:25])
        
        example_models = ", ".join(example_names)
        if len(org_models) > 3:
            example_models += "..."
        
        org_table.add_row(
            str(i),
            display_name,
            str(len(org_models)),
            example_models
        )
    
    console.print(org_table)
    
    if not show_all_orgs:
        console.print(f"\n[dim]💡 To see all {len(organizations)} organizations, type 'a'[/dim]")
    
    # Get organization choice
    while True:
        valid_choices = [str(i) for i in range(1, len(sorted_orgs) + 1)] + ["q"]
        if not show_all_orgs:
            valid_choices.append("a")
        
        choice = Prompt.ask(
            "\nSelect organization number, 'a' to show all organizations, or 'q' to quit" if not show_all_orgs else "\nSelect organization number (or 'q' to quit)",
            choices=valid_choices,
            show_choices=False
        )
        
        if choice.lower() == "q":
            console.print("👋 Goodbye!")
            return None
        elif choice.lower() == "a" and not show_all_orgs:
            console.print("Showing all organizations...\n")
            return interactive_organization_model_selection(show_all_orgs=True)
        
        selected_org, selected_models = sorted_orgs[int(choice) - 1]
        break
    
    # Step 2: Model Selection
    org_display_name = org_display_names.get(selected_org, selected_org.replace("-", " ").title())
    console.print(f"\n[bold cyan]🤖 Models from {org_display_name}:[/bold cyan]")
    
    terminal_width = console.size.width
    show_pricing = terminal_width >= 86
    show_image_cost = terminal_width >= 104
    show_context = terminal_width >= 114
    show_benchmark_date = terminal_width >= 74

    model_table = Table(
        show_header=True,
        header_style="bold cyan",
        pad_edge=False,
        collapse_padding=True,
    )
    # Keep selection index readable even in narrow terminals.
    model_table.add_column("No.", style="cyan", justify="right", width=3, min_width=3, max_width=3, no_wrap=True)
    model_table.add_column("Model Name", style="green", min_width=18 if show_pricing else 12, overflow="ellipsis")

    if show_pricing:
        model_table.add_column("In $/M", style="yellow", justify="right", width=8, min_width=6, no_wrap=True)
        model_table.add_column("Out $/M", style="orange3", justify="right", width=8, min_width=6, no_wrap=True)
    if show_image_cost:
        model_table.add_column("Img $/K", style="magenta", justify="right", width=8, min_width=6, no_wrap=True)
    if show_context:
        model_table.add_column("Ctx", style="blue", justify="right", width=8, min_width=5, no_wrap=True)
    if show_benchmark_date:
        model_table.add_column("Benchmarked", style="cyan", justify="center", width=11, min_width=8, no_wrap=True)

    if not show_context or not show_image_cost or not show_pricing or not show_benchmark_date:
        console.print(
            "[dim]Compact view enabled for terminal width: prioritizing selection number + model name.[/dim]"
        )
    
    for i, model in enumerate(selected_models, 1):
        model_id = model.get("id", "")
        model_name = model.get("name", model.get("id", "Unknown"))
        context_length = model.get("context_length", "Unknown")
        pricing = model.get("pricing", {})
        
        # Format context length
        if isinstance(context_length, int):
            context_str = f"{context_length:,}"
        else:
            context_str = str(context_length)
        
        # Format pricing
        try:
            input_cost = pricing.get("prompt", "0")
            output_cost = pricing.get("completion", "0")
            image_cost = pricing.get("image", "0")
            
            input_per_m = f"${float(input_cost) * 1_000_000:.2f}" if input_cost != "0" else "Free"
            output_per_m = f"${float(output_cost) * 1_000_000:.2f}" if output_cost != "0" else "Free"
            image_per_k = f"${float(image_cost) * 1_000:.2f}" if image_cost != "0" else "-"
        except (ValueError, TypeError):
            input_per_m = "Unknown"
            output_per_m = "Unknown"
            image_per_k = "Unknown"
        
        row_cells = [str(i), model_name]
        if show_pricing:
            row_cells.extend([input_per_m, output_per_m])
        if show_image_cost:
            row_cells.append(image_per_k)
        if show_context:
            row_cells.append(context_str)
        if show_benchmark_date:
            row_cells.append(last_benchmarked_lookup.get(model_id, "Never"))

        model_table.add_row(*row_cells)
    
    console.print(model_table)
    
    # Get model choice
    while True:
        choice = Prompt.ask(
            "\nSelect model number, 'b' to choose different organization, or 'q' to quit",
            choices=[str(i) for i in range(1, len(selected_models) + 1)] + ["b", "q"],
            show_choices=False
        )
        
        if choice.lower() == "q":
            console.print("👋 Goodbye!")
            return None
        elif choice.lower() == "b":
            return interactive_organization_model_selection(show_all_orgs)
        
        selected_model = selected_models[int(choice) - 1]
        break
    
    # Step 3: Provider Policy Review
    model_id = selected_model.get("id", "")
    console.print(f"\n✅ Selected: [bold green]{selected_model.get('name', model_id)}[/bold green]")
    console.print(f"[dim]Model ID: {model_id}[/dim]")
    
    # Show provider policies and get confirmation  
    providers_dict = {"data": providers_data} if providers_data else None
    if not display_provider_policies(model_id, providers_dict):
        console.print("[yellow]⚠️  Model selection cancelled by user[/yellow]")
        return interactive_organization_model_selection(show_all_orgs)
    
    # Show usage example
    console.print("\n[bold blue]💡 To run this model:[/bold blue]")
    console.print(f"uv run openrouter-inference --model [green]{model_id}[/green]")
    console.print(f"uv run openrouter-inference --model [green]{model_id}[/green] [blue]--interactive[/blue]")
    
    return model_id


# Keep the legacy function name for backwards compatibility
def interactive_provider_model_selection(show_all_providers: bool = False):
    """Legacy function name - redirects to organization-based selection."""
    return interactive_organization_model_selection(show_all_orgs=show_all_providers)


def interactive_config_prompt(model_name: str) -> Dict[str, Any]:
    """Interactive prompt for model configuration when not provided via CLI."""
    
    console.print("\n[bold blue]🔧 Model Configuration[/bold blue]")
    console.print(f"Setting up configuration for: [bold green]{model_name}[/bold green]\n")
    
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
