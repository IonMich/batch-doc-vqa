#!/usr/bin/env python3
"""
Gemini CLI for batch document VQA.
"""
import argparse
from .inference import run_gemini_inference
from .ui import interactive_model_selection


def parse_pages(raw_value: str | None) -> list[int]:
    """Parse comma-separated page numbers."""
    if not raw_value:
        return [1, 3]

    pages: list[int] = []
    seen: set[int] = set()
    for chunk in raw_value.split(","):
        value = chunk.strip()
        if not value:
            continue
        try:
            page = int(value)
        except ValueError as exc:
            raise ValueError(f"Invalid page value: {value!r}") from exc
        if page <= 0 or page in seen:
            continue
        seen.add(page)
        pages.append(page)

    if not pages:
        raise ValueError("At least one valid page is required")
    return pages


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
    parser.add_argument("--images-dir", type=str, default="imgs/q11", help="Directory containing document page images (default: imgs/q11)")
    parser.add_argument("--doc-info", type=str, help="Optional doc_info.csv path. If provided, image list is derived from this CSV.")
    parser.add_argument("--pages", type=str, default="1,3", help="Comma-separated page numbers to include (default: 1,3)")
    
    args = parser.parse_args()
    
    # Interactive model selection if no model specified
    if not args.model:
        selected_model = interactive_model_selection()
        if not selected_model:
            return
        args.model = selected_model
    
    try:
        pages = parse_pages(args.pages)
    except ValueError as exc:
        parser.error(str(exc))
        return

    # Run inference
    run_name = run_gemini_inference(
        model_id=args.model, 
        temperature=args.temperature,
        model_size=args.model_size,
        open_weights=args.open_weights,
        license_info=args.license_info,
        interactive=args.interactive,
        images_dir=args.images_dir,
        doc_info_file=args.doc_info,
        pages=pages,
    )
    
    print(f"\nRun completed: {run_name}")
    print("Generate benchmark table with:")
    print(f"uv run generate-benchmark-table --patterns google")


if __name__ == "__main__":
    main()
