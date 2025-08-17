#!/usr/bin/env python3
"""
Gemini interactive user interface components.
"""
import os
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.table import Table

console = Console()


def fetch_gemini_models() -> Optional[List[Dict[str, Any]]]:
    """Fetch available Gemini models from the API. Returns None if unable to fetch."""
    try:
        from google import genai
        
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return None
        
        client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
        
        # Fetch models from API
        models_response = client.models.list()
        models = []
        
        for model in models_response:
            # Convert model to dict for consistent access
            if hasattr(model, 'model_dump'):
                model_dict = model.model_dump()
            elif hasattr(model, '__dict__'):
                model_dict = model.__dict__
            else:
                model_dict = dict(model) if hasattr(model, 'items') else {}
            
            # Filter for vision-capable models (models that support image input)
            # Since there's no explicit API field, we use naming patterns and known capabilities
            model_name = model_dict.get("name", "").lower()
            
            # Skip text-only models
            if any(skip in model_name for skip in ["embedding", "text-embedding", "aqa"]):
                continue
                
            # Include models we know support vision
            if any(vision_model in model_name for vision_model in [
                "gemini-1.5", "gemini-2.0", "gemini-2.5", "gemini-pro-vision"
            ]):
                models.append({
                    "id": model_dict.get("name", "").replace("models/", ""),
                    "name": model_dict.get("display_name", model_dict.get("name", "Unknown")),
                    "description": model_dict.get("description", "Vision-capable Gemini model"),
                    "supported_generation_methods": model_dict.get("supported_generation_methods", []),
                    "input_token_limit": model_dict.get("input_token_limit"),
                    "output_token_limit": model_dict.get("output_token_limit"),
                })
        
        # Sort by name for consistent display
        models.sort(key=lambda x: x["name"])
        
        return models if models else None
        
    except Exception as e:
        console.print(f"[red]❌ Could not fetch models from Gemini API: {e}[/red]")
        return None


def check_api_key_setup() -> bool:
    """Check if Gemini API key is properly configured."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("\n[yellow]⚠️  Gemini API key not found in environment or .env file[/yellow]")
        console.print("\n[cyan]You can get your API key from: https://ai.google.dev/api[/cyan]")
        
        if Confirm.ask("\nWould you like to enter your API key now?"):
            api_key = Prompt.ask("Enter your Gemini API key", password=True)
            if api_key:
                # Offer to save to .env file
                if Confirm.ask("Save this key to .env file for future use?"):
                    try:
                        with open('.env', 'a') as f:
                            f.write(f"\nGEMINI_API_KEY={api_key}\n")
                        console.print("[green]✅ API key saved to .env file[/green]")
                        # Set for current session
                        os.environ['GEMINI_API_KEY'] = api_key
                        return True
                    except Exception as e:
                        console.print(f"[red]❌ Failed to save to .env: {e}[/red]")
                        console.print("[yellow]Setting for current session only[/yellow]")
                        os.environ['GEMINI_API_KEY'] = api_key
                        return True
                else:
                    # Set for current session only
                    os.environ['GEMINI_API_KEY'] = api_key
                    return True
            else:
                console.print("[yellow]⚠️  Continuing without API key - inference will fail[/yellow]")
                return False
        else:
            console.print("[yellow]⚠️  Continuing without API key - inference will fail[/yellow]")
            return False
    return True


def interactive_model_selection() -> Optional[str]:
    """Interactive model selection for Gemini models."""
    console.print("\n[bold blue]🤖 Google Gemini Model Selection[/bold blue]")
    
    # Check API key setup
    if not check_api_key_setup():
        return None
    
    # Fetch available models
    console.print("\n[dim]🔍 Fetching available models from Gemini API...[/dim]")
    models = fetch_gemini_models()
    
    if not models:
        console.print("\n[red]❌ Unable to fetch models from Gemini API[/red]")
        console.print("[yellow]Please check your API key and internet connection[/yellow]")
        console.print("[dim]You can still use gemini-inference with a specific --model parameter[/dim]")
        return None
    
    console.print(f"\n[green]✅ Found {len(models)} available Gemini models[/green]")
    
    # Display model selection table
    console.print("\n[bold cyan]📋 Select a Gemini Model:[/bold cyan]")
    
    model_table = Table(show_header=True, header_style="bold cyan")
    model_table.add_column("No.", style="cyan", width=3)
    model_table.add_column("Model Name", style="green", width=35)
    model_table.add_column("Input Limit", style="blue", justify="right", width=12)
    model_table.add_column("Output Limit", style="magenta", justify="right", width=12)
    model_table.add_column("Description", style="white", width=50)
    
    for i, model in enumerate(models, 1):
        input_limit = model.get('input_token_limit')
        output_limit = model.get('output_token_limit')
        
        input_str = f"{input_limit:,}" if isinstance(input_limit, int) else str(input_limit) if input_limit else "Unknown"
        output_str = f"{output_limit:,}" if isinstance(output_limit, int) else str(output_limit) if output_limit else "Unknown"
        
        # Truncate long descriptions
        description = model.get("description", "Vision-capable Gemini model")
        if len(description) > 50:
            description = description[:47] + "..."
        
        model_table.add_row(
            str(i),
            model["name"],
            input_str,
            output_str,
            description
        )
    
    console.print(model_table)
    
    # Get model choice
    while True:
        choice = Prompt.ask(
            "\nSelect model number or 'q' to quit",
            choices=[str(i) for i in range(1, len(models) + 1)] + ["q"],
            show_choices=False
        )
        
        if choice.lower() == "q":
            console.print("👋 Goodbye!")
            return None
        
        selected_model = models[int(choice) - 1]
        break
    
    # Display selection confirmation
    model_id = selected_model["id"]
    console.print(f"\n✅ Selected: [bold green]{selected_model['name']}[/bold green]")
    console.print(f"[dim]Model ID: {model_id}[/dim]")
    
    # Show privacy notice
    console.print("\n[bold cyan]🔒 Privacy Notice[/bold cyan]")
    console.print("[dim]This model will send your images to Google's Gemini API.[/dim]")
    console.print("[dim]Review Google's privacy policy: https://ai.google.dev/gemini-api/terms[/dim]")
    
    if not Confirm.ask(f"\nProceed with {selected_model['name']}?", default=True):
        console.print("[yellow]⚠️  Model selection cancelled by user[/yellow]")
        return interactive_model_selection()  # Allow re-selection
    
    # Show usage example
    console.print("\n[bold blue]💡 To run this model:[/bold blue]")
    console.print(f"uv run --extra gemini gemini-inference --model [green]{model_id}[/green]")
    console.print(f"uv run --extra gemini gemini-inference --model [green]{model_id}[/green] [blue]--interactive[/blue]")
    
    return model_id


def get_model_pricing(model_name: str) -> Dict[str, float]:
    """Get pricing for a Gemini model via user input."""
    console.print(f"\n[bold yellow]💰 Pricing Configuration for {model_name}[/bold yellow]")
    console.print("[dim]Enter pricing in USD per 1 million tokens[/dim]")
    console.print("[dim]Reference: https://ai.google.dev/gemini-api/docs/pricing[/dim]")
    
    # Show reference pricing for common models
    reference_pricing = {
        "gemini-2.5-flash": {"input": 0.30, "output": 2.50},
        "gemini-2.5-flash-lite": {"input": 0.10, "output": 0.40},
        "gemini-2.5-pro": {"input": 1.25, "output": 10.00},  # ≤200k tokens
    }
    
    if model_name in reference_pricing:
        ref = reference_pricing[model_name]
        console.print(f"[dim]Reference for {model_name}: Input ${ref['input']}, Output ${ref['output']} per 1M tokens[/dim]")
        if "pro" in model_name.lower():
            console.print(f"[dim]Note: Pro pricing varies by context length (${ref['input']}-$2.50 input, ${ref['output']}-$15.00 output)[/dim]")
    
    while True:
        try:
            input_price = float(Prompt.ask("Input token price ($ per 1M tokens)"))
            output_price = float(Prompt.ask("Output token price ($ per 1M tokens)"))
            
            console.print(f"\n[cyan]Pricing summary:[/cyan]")
            console.print(f"  Input: ${input_price:.3f} per 1M tokens")
            console.print(f"  Output: ${output_price:.3f} per 1M tokens")
            
            if Confirm.ask("Confirm these prices?", default=True):
                return {
                    "input_per_million": input_price,
                    "output_per_million": output_price,
                }
        except ValueError:
            console.print("[red]Please enter valid decimal numbers[/red]")


def interactive_config_prompt(model_name: str) -> Dict[str, Any]:
    """Interactive prompt for model configuration when not provided via CLI."""
    
    console.print("\n[bold blue]🔧 Model Configuration[/bold blue]")
    console.print(f"Setting up configuration for: [bold green]{model_name}[/bold green]\n")
    
    # For Gemini models, we know they're proprietary from Google
    console.print("[dim]Note: All Gemini models are proprietary Google models[/dim]")
    
    # Model size - Google doesn't publish exact parameter counts
    size_options = ["Unknown", "Small", "Medium", "Large", "Custom"]
    size_table = Table(show_header=False, box=None)
    for i, size in enumerate(size_options, 1):
        size_table.add_row(f"[cyan]{i}[/cyan]", size)
    
    console.print("\n[bold]Model Size Category:[/bold]")
    console.print(size_table)
    size_choice = Prompt.ask(
        "Choose model size category", 
        choices=[str(i) for i in range(1, len(size_options) + 1)],
        default="1"
    )
    model_size = size_options[int(size_choice) - 1]
    
    if model_size == "Custom":
        model_size = Prompt.ask("Enter custom model size description")
    
    # Get pricing information
    pricing = get_model_pricing(model_name)
    
    # Gemini models are always proprietary
    open_weights = False
    license_info = "Proprietary (Google)"
    
    console.print(f"\n[green]✅ Configuration complete:[/green]")
    console.print(f"  Model size: {model_size}")
    console.print(f"  Open weights: {open_weights}")
    console.print(f"  License: {license_info}")
    console.print(f"  Input pricing: ${pricing['input_per_million']:.3f}/1M tokens")
    console.print(f"  Output pricing: ${pricing['output_per_million']:.3f}/1M tokens")
    
    return {
        "model_size": model_size,
        "open_weights": open_weights, 
        "license_info": license_info,
        "pricing": pricing,
    }