#!/usr/bin/env python3
"""
Combined script to update both BENCHMARKS.md and README.md benchmark sections.
"""
import subprocess
import sys
import argparse
import filecmp
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from batch_doc_vqa.benchmarks.pareto_plot import PARETO_FINGERPRINT_KEY


def _artifacts_match(generated: Path, committed: Path) -> bool:
    """Compare text exactly and PNG plots by their platform-neutral data fingerprint."""
    if not committed.exists():
        return False
    if generated.suffix.lower() != ".png" or committed.suffix.lower() != ".png":
        return filecmp.cmp(generated, committed, shallow=False)
    try:
        with Image.open(generated) as generated_image, Image.open(committed) as committed_image:
            generated_fingerprint = generated_image.info.get(PARETO_FINGERPRINT_KEY)
            committed_fingerprint = committed_image.info.get(PARETO_FINGERPRINT_KEY)
            return (
                isinstance(generated_fingerprint, str)
                and generated_fingerprint == committed_fingerprint
                and generated_image.size == committed_image.size
                and generated_image.mode == committed_image.mode
            )
    except (OSError, ValueError):
        return False

def run_command(command, description, interactive=False):
    """Run a command and handle errors."""
    print(f"📊 {description}...")
    try:
        if interactive:
            # For interactive commands, don't capture output so prompts work
            result = subprocess.run(command, shell=True, check=True)
            return True
        else:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
            if result.stdout.strip():
                print(result.stdout)
            return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error {description.lower()}: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Update both BENCHMARKS.md and README.md")
    parser.add_argument("--patterns", nargs="*", help="Patterns to filter runs (e.g., 'glm' 'qwen')")
    parser.add_argument("--source", choices=("auto", "local", "published"), default="auto",
                        help="Use the finalized published archive when available (default: auto).")
    parser.add_argument("--published-runs-dir", default="benchmarks/published/q11/runs",
                        help="Directory containing sanitized published run summaries.")
    parser.add_argument("--no-interactive", action="store_true", 
                       help="Skip interactive model review (add unknown models to needs_review list)")
    parser.add_argument("--interactive", action="store_true",
                       help="Force interactive model review even if unknown models exist")
    parser.add_argument(
        "--pareto-label-mode",
        choices=("frontier", "none", "all"),
        default="frontier",
        help="Which model names to print in the Pareto plot (default: frontier).",
    )
    parser.add_argument(
        "--hide-non-frontier-labels",
        action="store_true",
        help="Deprecated alias for --pareto-label-mode frontier.",
    )
    parser.add_argument(
        "--extra-id-lev-pareto",
        action="store_true",
        help="Generate an additional Pareto plot with ID Avg d_Lev vs total cost (non-default)",
    )
    parser.add_argument(
        "--id-lev-output",
        default="pareto_plot_id_lev.png",
        help="Output path for optional ID Avg d_Lev Pareto plot",
    )
    parser.add_argument(
        "--interactive-pareto-output",
        default="docs/pareto.html",
        help="Output path for the standalone interactive Pareto plot",
    )
    parser.add_argument(
        "--no-interactive-pareto",
        action="store_true",
        help="Skip generating the standalone interactive Pareto plot",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Regenerate into temporary files and fail if committed benchmark artifacts are stale.",
    )
    args = parser.parse_args()

    check_dir = None
    benchmarks_output = "BENCHMARKS.md"
    pareto_output = "pareto_plot.png"
    id_lev_output = args.id_lev_output
    interactive_output = args.interactive_pareto_output
    readme_output = "README.md"
    if args.check:
        check_dir = Path(tempfile.mkdtemp(prefix="batch-doc-vqa-benchmarks-"))
        benchmarks_output = str(check_dir / "BENCHMARKS.md")
        pareto_output = str(check_dir / "pareto_plot.png")
        id_lev_output = str(check_dir / "pareto_plot_id_lev.png")
        interactive_output = str(check_dir / "pareto.html")
        readme_output = str(check_dir / "README.md")
        shutil.copyfile("README.md", readme_output)
    
    # Build command for benchmark table generation
    cmd_parts = ["uv", "run", "generate-benchmark-table", "--format", "markdown", "--output", benchmarks_output]
    cmd_parts.extend(["--source", args.source, "--published-runs-dir", args.published_runs_dir])
    if args.check:
        cmd_parts.append("--no-cache")
    
    if args.patterns:
        cmd_parts.extend(["--patterns"] + args.patterns)
    
    # Handle interactive mode
    if args.no_interactive:
        cmd_parts.append("--no-interactive")
    elif not args.interactive:
        # Default behavior: let generate-benchmark-table handle interactivity
        # It will prompt if there are unknown models
        pass
    
    benchmarks_cmd = " ".join(cmd_parts)
    
    # Update BENCHMARKS.md - use interactive mode unless --no-interactive specified
    is_interactive = not args.no_interactive
    if not run_command(benchmarks_cmd, "Updating BENCHMARKS.md", interactive=is_interactive):
        sys.exit(1)
    
    # Generate Pareto plot
    pareto_cmd_parts = ["uv", "run", "generate-pareto-plot", "--output", pareto_output]
    pareto_cmd_parts.extend(["--source", args.source, "--published-runs-dir", args.published_runs_dir])
    if args.check:
        pareto_cmd_parts.append("--no-cache")
    if args.patterns:
        pareto_cmd_parts.extend(["--patterns"] + args.patterns)
    if args.no_interactive:
        pareto_cmd_parts.append("--no-interactive")
    pareto_label_mode = "frontier" if args.hide_non_frontier_labels else args.pareto_label_mode
    pareto_cmd_parts.extend(["--label-mode", pareto_label_mode])
    if args.extra_id_lev_pareto:
        pareto_cmd_parts.append("--extra-id-lev-pareto")
        pareto_cmd_parts.extend(["--id-lev-output", id_lev_output])
    
    pareto_cmd = " ".join(pareto_cmd_parts)
    if not run_command(pareto_cmd, "Generating Pareto plot", interactive=is_interactive):
        sys.exit(1)

    if not args.no_interactive_pareto:
        interactive_pareto_cmd_parts = [
            "uv",
            "run",
            "generate-interactive-pareto",
            "--output",
            interactive_output,
        ]
        interactive_pareto_cmd_parts.extend(["--source", args.source, "--published-runs-dir", args.published_runs_dir])
        if args.check:
            interactive_pareto_cmd_parts.append("--no-cache")
        if args.patterns:
            interactive_pareto_cmd_parts.extend(["--patterns"] + args.patterns)
        if args.no_interactive:
            interactive_pareto_cmd_parts.append("--no-interactive")
        interactive_pareto_cmd = " ".join(interactive_pareto_cmd_parts)
        if not run_command(
            interactive_pareto_cmd,
            "Generating interactive Pareto plot",
            interactive=is_interactive,
        ):
            sys.exit(1)
    
    # Update README.md
    readme_cmd = (
        "uv run update-readme-section"
        f" --readme {readme_output}"
        f" --source {args.source}"
        f" --published-runs-dir {args.published_runs_dir}"
    )
    if args.check:
        readme_cmd += " --no-cache"
    if not run_command(readme_cmd, "Updating README.md", interactive=False):
        sys.exit(1)

    if args.check:
        expected_outputs = [
            (Path(benchmarks_output), Path("BENCHMARKS.md")),
            (Path(pareto_output), Path("pareto_plot.png")),
            (Path(readme_output), Path("README.md")),
        ]
        if not args.no_interactive_pareto:
            expected_outputs.append((Path(interactive_output), Path(args.interactive_pareto_output)))
        if args.extra_id_lev_pareto:
            expected_outputs.append((Path(id_lev_output), Path(args.id_lev_output)))
        stale = [
            str(committed)
            for generated, committed in expected_outputs
            if not _artifacts_match(generated, committed)
        ]
        shutil.rmtree(check_dir, ignore_errors=True)
        if stale:
            print("Generated benchmark artifacts are stale:")
            for path in stale:
                print(f"  - {path}")
            if os.environ.get("GITHUB_ACTIONS") == "true":
                print(f"::error title=Stale benchmark artifacts::{', '.join(stale)}")
            sys.exit(1)
        print("✅ Generated benchmark artifacts are current.")
        return
    
    generated_plot_label = "Pareto plots" if args.extra_id_lev_pareto else "Pareto plot"
    print(f"\n✅ Successfully updated BENCHMARKS.md, README.md, and {generated_plot_label}!")
    print("\n💡 To commit and push changes, run:")
    files_to_add = ["BENCHMARKS.md", "README.md", "pareto_plot.png"]
    if not args.no_interactive_pareto:
        files_to_add.append(args.interactive_pareto_output)
    if args.extra_id_lev_pareto:
        files_to_add.append(args.id_lev_output)
    print(f"   git add {' '.join(files_to_add)}")
    print("   git commit -m 'Update benchmark tables and Pareto plot'")
    print("   git push")
    print("\n💡 Usage notes:")
    print("   • Default: Will prompt for interactive model review if unknown models found")
    print("   • --no-interactive: Skip review, add unknown models to needs_review list") 
    print("   • --interactive: Force review even if no unknown models")
    print("   • --pareto-label-mode: Show labels for frontier points (default), none, or all models")
    print("   • --interactive-pareto-output: Write the standalone interactive Pareto plot (default: docs/pareto.html)")
    print("   • --no-interactive-pareto: Skip the standalone interactive Pareto plot")
    print("   • --extra-id-lev-pareto: Also generate pareto_plot_id_lev.png with inverted d_Lev y-axis")

if __name__ == "__main__":
    main()
