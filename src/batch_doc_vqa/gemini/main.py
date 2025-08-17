#!/usr/bin/env python3
"""
Gemini main CLI entry point with interactive UI.
"""
import argparse
from .ui import interactive_model_selection
from .inference import run_gemini_inference


def main():
    """Main CLI entry point for Gemini UI."""
    parser = argparse.ArgumentParser(
        description="Interactive Gemini model selection and inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive model selection
  uv run --extra gemini gemini-ui
  
  # Direct inference (bypass UI)
  uv run --extra gemini gemini-inference --model gemini-2.5-flash
        """
    )
    
    parser.add_argument("--model", type=str, help="Skip UI and run inference with specific model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature (default: 0.0)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Use interactive prompts for missing configuration")
    
    args = parser.parse_args()
    
    if args.model:
        # Direct inference mode
        run_name = run_gemini_inference(
            model_id=args.model,
            temperature=args.temperature,
            interactive=args.interactive
        )
        print(f"\nRun completed: {run_name}")
    else:
        # Interactive UI mode
        selected_model = interactive_model_selection()
        if selected_model:
            run_name = run_gemini_inference(
                model_id=selected_model,
                temperature=args.temperature,
                interactive=args.interactive
            )
            print(f"\nRun completed: {run_name}")
            print("Generate benchmark table with:")
            print("uv run generate-benchmark-table --patterns google")


if __name__ == "__main__":
    main()