#!/usr/bin/env python3
"""
Gemini CLI for batch document VQA.
"""
import argparse
from .inference import run_gemini_inference
from .ui import interactive_model_selection


def main():
    """Main CLI entry point for Gemini inference."""
    parser = argparse.ArgumentParser(
        description="Run inference using Gemini vision models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default model
  uv run --extra gemini gemini-inference
  
  # Run with specific model
  uv run --extra gemini gemini-inference --model gemini-2.0-flash-exp
  
  # Run with different temperature
  uv run --extra gemini gemini-inference --temperature 0.1
        """
    )
    
    parser.add_argument("--model", type=str,
                       help="Gemini model name (if not specified, interactive selection)")
    parser.add_argument("--temperature", type=float, default=0.0,
                       help="Temperature (default: 0.0 for deterministic output)")
    parser.add_argument("--model-size", type=str, help="Model size (for metadata)")
    parser.add_argument("--open-weights", action="store_true", help="Mark as open-weights model")
    parser.add_argument("--license-info", type=str, help="License information")
    parser.add_argument("--interactive", "-i", action="store_true", help="Use interactive prompts for missing configuration")
    
    args = parser.parse_args()
    
    # Interactive model selection if no model specified
    if not args.model:
        selected_model = interactive_model_selection()
        if not selected_model:
            return
        args.model = selected_model
    
    # Run inference
    run_name = run_gemini_inference(
        model_id=args.model, 
        temperature=args.temperature,
        model_size=args.model_size,
        open_weights=args.open_weights,
        license_info=args.license_info,
        interactive=args.interactive
    )
    
    print(f"\nRun completed: {run_name}")
    print("Generate benchmark table with:")
    print(f"uv run generate-benchmark-table --patterns google")


if __name__ == "__main__":
    main()