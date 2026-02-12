#!/usr/bin/env python3
"""
OpenRouter CLI utilities and main entry point.
"""
import os
import argparse
import dotenv

from rich.console import Console
from rich.prompt import Prompt, Confirm

dotenv.load_dotenv()

console = Console()


def parse_provider_order(raw_value: str | None) -> list[str]:
    """Parse comma-separated provider slugs into a clean ordered list."""
    if not raw_value:
        return []

    providers: list[str] = []
    seen: set[str] = set()
    for chunk in raw_value.split(","):
        slug = chunk.strip().lower()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        providers.append(slug)
    return providers


# Import shared utilities from core
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
  uv run openrouter-inference --model z-ai/glm-4.5v
  
  # List models with special configurations
  uv run openrouter-inference --list-overrides
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
    parser.add_argument("--repetition-penalty", type=float, help="Repetition penalty to apply (higher values discourage loops)")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of parallel requests (default: 1)")
    parser.add_argument("--rate-limit", type=float, default=None, help="Max requests per second across all threads (optional)")
    parser.add_argument("--retry-max", type=int, default=3, help="Max retries per image for 5xx errors (default: 3)")
    parser.add_argument("--retry-base-delay", type=float, default=2.0, help="Base delay (seconds) for exponential backoff on 5xx (default: 2.0)")
    parser.add_argument(
        "--provider-order",
        type=str,
        help="Comma-separated OpenRouter provider slugs (preferred order, e.g. deepinfra,crusoe)",
    )
    parser.add_argument(
        "--no-fallbacks",
        action="store_true",
        help="Disable provider fallbacks (pairs best with --provider-order)",
    )
    parser.add_argument(
        "--provider-sort",
        choices=["price", "throughput", "latency"],
        help="Provider sorting preference for OpenRouter routing",
    )
    
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

    provider_order = parse_provider_order(args.provider_order)
    allow_fallbacks = False if args.no_fallbacks else None

    from .inference import run_openrouter_inference
    run_name = run_openrouter_inference(
        model_name=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        model_size=args.model_size,
        open_weights=args.open_weights,
        license_info=args.license_info,
        interactive=args.interactive,
        concurrency=args.concurrency,
        rate_limit=args.rate_limit,
        retry_max=args.retry_max,
        retry_base_delay=args.retry_base_delay,
        provider_order=provider_order if provider_order else None,
        provider_allow_fallbacks=allow_fallbacks,
        provider_sort=args.provider_sort,
    )
    
    print(f"\nRun completed: {run_name}")
    print("Generate benchmark table with:")
    print(f"uv run generate-benchmark-table --patterns {args.model.split('/')[0]}")


if __name__ == "__main__":
    main()
