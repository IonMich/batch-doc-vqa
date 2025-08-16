#!/usr/bin/env python3
"""
OpenRouter CLI utilities and main entry point.
"""
import os
import re
import base64
import argparse
import dotenv

from rich.console import Console
from rich.prompt import Prompt, Confirm

dotenv.load_dotenv()

console = Console()


def filepath_to_base64(filepath):
    """Convert file to base64 encoded string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_imagepaths(folder, pattern):
    """Get image paths matching pattern from folder."""
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if re.match(pattern, file):
                images.append(os.path.join(root, file))
    # sort by integers in the filename
    images.sort(key=natural_sort_key)
    return images


def natural_sort_key(s):
    """Natural sorting key for filenames with numbers."""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


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


def list_model_overrides():
    """List models with special configuration overrides."""
    from .api import MODEL_CONFIG_OVERRIDES
    
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


def setup_api_key():
    """Setup OpenRouter API key interactively."""
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        console.print("\n[yellow]⚠️  OpenRouter API key not found in environment or .env file[/yellow]")
        console.print("\n[dim]💡 Without an API key, only free models with limited requests are available[/dim]")
        console.print("\n[cyan]You can get your API key from: https://openrouter.ai/keys[/cyan]")
        
        if Confirm.ask("\nWould you like to enter your API key now for access to all models?"):
            api_key = Prompt.ask("Enter your OpenRouter API key", password=True)
            if api_key:
                # Offer to save to .env file
                if Confirm.ask("Save this key to .env file for future use?"):
                    try:
                        with open('.env', 'a') as f:
                            f.write(f"\nOPENROUTER_API_KEY={api_key}\n")
                        console.print("[green]✅ API key saved to .env file[/green]")
                        # Set for current session
                        os.environ['OPENROUTER_API_KEY'] = api_key
                    except Exception as e:
                        console.print(f"[red]❌ Failed to save to .env: {e}[/red]")
                        console.print("[yellow]Setting for current session only[/yellow]")
                        os.environ['OPENROUTER_API_KEY'] = api_key
                else:
                    # Set for current session only
                    os.environ['OPENROUTER_API_KEY'] = api_key
            else:
                console.print("[yellow]⚠️  Continuing without API key - only free models available[/yellow]")
        else:
            console.print("[yellow]⚠️  Continuing without API key - only free models available[/yellow]")


def main():
    """Main CLI entry point."""
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
    
    # Check for API key first (before showing provider selection)
    setup_api_key()
    
    if not args.model:
        # Interactive provider and model selection
        from .ui import interactive_provider_model_selection
        selected_model = interactive_provider_model_selection()
        if not selected_model:
            return
        # Set the selected model for inference
        args.model = selected_model
    
    from .inference import run_openrouter_inference
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