#!/usr/bin/env python3
"""
Combined script to update both BENCHMARKS.md and README.md benchmark sections.
"""
import subprocess
import sys
import argparse
from pathlib import Path

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
    parser.add_argument("--no-interactive", action="store_true", 
                       help="Skip interactive model review (add unknown models to needs_review list)")
    parser.add_argument("--interactive", action="store_true",
                       help="Force interactive model review even if unknown models exist")
    parser.add_argument("--hide-non-frontier-labels", action="store_true",
                       help="Hide labels for non-frontier models in Pareto plot (default: show all labels in gray)")
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
    args = parser.parse_args()
    
    # Build command for benchmark table generation
    cmd_parts = ["uv", "run", "generate-benchmark-table", "--format", "markdown", "--output", "BENCHMARKS.md"]
    
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
    pareto_cmd_parts = ["uv", "run", "generate-pareto-plot", "--output", "pareto_plot.png"]
    if args.patterns:
        pareto_cmd_parts.extend(["--patterns"] + args.patterns)
    if args.no_interactive:
        pareto_cmd_parts.append("--no-interactive")
    if args.hide_non_frontier_labels:
        pareto_cmd_parts.append("--hide-non-frontier-labels")
    if args.extra_id_lev_pareto:
        pareto_cmd_parts.append("--extra-id-lev-pareto")
        pareto_cmd_parts.extend(["--id-lev-output", args.id_lev_output])
    
    pareto_cmd = " ".join(pareto_cmd_parts)
    if not run_command(pareto_cmd, "Generating Pareto plot", interactive=is_interactive):
        sys.exit(1)
    
    # Update README.md
    readme_cmd = "uv run update-readme-section"
    if not run_command(readme_cmd, "Updating README.md", interactive=False):
        sys.exit(1)
    
    generated_plot_label = "Pareto plots" if args.extra_id_lev_pareto else "Pareto plot"
    print(f"\n✅ Successfully updated BENCHMARKS.md, README.md, and {generated_plot_label}!")
    print("\n💡 To commit and push changes, run:")
    files_to_add = ["BENCHMARKS.md", "README.md", "pareto_plot.png"]
    if args.extra_id_lev_pareto:
        files_to_add.append(args.id_lev_output)
    print(f"   git add {' '.join(files_to_add)}")
    print("   git commit -m 'Update benchmark tables and Pareto plot'")
    print("   git push")
    print("\n💡 Usage notes:")
    print("   • Default: Will prompt for interactive model review if unknown models found")
    print("   • --no-interactive: Skip review, add unknown models to needs_review list") 
    print("   • --interactive: Force review even if no unknown models")
    print("   • --hide-non-frontier-labels: Hide model names for non-frontier points (default: show in gray)")
    print("   • --extra-id-lev-pareto: Also generate pareto_plot_id_lev.png with inverted d_Lev y-axis")

if __name__ == "__main__":
    main()
