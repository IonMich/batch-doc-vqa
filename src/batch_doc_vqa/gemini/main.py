#!/usr/bin/env python3
"""
Gemini main CLI entry point with interactive UI.
"""
import argparse
from .ui import interactive_model_selection
from .inference import run_gemini_inference


def parse_pages(raw_value: str | None) -> list[int]:
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
    parser.add_argument("--images-dir", type=str, default="imgs/q11", help="Directory containing document page images (default: imgs/q11)")
    parser.add_argument("--doc-info", type=str, help="Optional doc_info.csv path. If provided, image list is derived from this CSV.")
    parser.add_argument("--pages", type=str, default="1,3", help="Comma-separated page numbers to include (default: 1,3)")
    
    args = parser.parse_args()
    
    try:
        pages = parse_pages(args.pages)
    except ValueError as exc:
        parser.error(str(exc))
        return

    if args.model:
        # Direct inference mode
        run_name = run_gemini_inference(
            model_id=args.model,
            temperature=args.temperature,
            interactive=args.interactive,
            images_dir=args.images_dir,
            doc_info_file=args.doc_info,
            pages=pages,
        )
        print(f"\nRun completed: {run_name}")
    else:
        # Interactive UI mode
        selected_model = interactive_model_selection()
        if selected_model:
            run_name = run_gemini_inference(
                model_id=selected_model,
                temperature=args.temperature,
                interactive=args.interactive,
                images_dir=args.images_dir,
                doc_info_file=args.doc_info,
                pages=pages,
            )
            print(f"\nRun completed: {run_name}")
            print("Generate benchmark table with:")
            print("uv run generate-benchmark-table --patterns google")


if __name__ == "__main__":
    main()
