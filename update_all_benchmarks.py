#!/usr/bin/env python3
"""
Combined script to update both BENCHMARKS.md and README.md benchmark sections.
"""
import subprocess
import sys
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"📊 {description}...")
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, check=True)
        if result.stdout.strip():
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error {description.lower()}: {e}")
        if e.stderr:
            print(f"Error details: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Update both BENCHMARKS.md and README.md")
    parser.add_argument("--patterns", nargs="*", help="Patterns to filter runs (e.g., 'glm' 'qwen')")
    parser.add_argument("--no-interactive", action="store_true", 
                       help="Skip interactive model review")
    args = parser.parse_args()
    
    # Build command for benchmark table generation
    cmd_parts = ["python", "generate_benchmark_table.py", "--format", "markdown", "--output", "BENCHMARKS.md"]
    
    if args.patterns:
        cmd_parts.extend(["--patterns"] + args.patterns)
    
    if args.no_interactive:
        cmd_parts.append("--no-interactive")
    
    benchmarks_cmd = " ".join(cmd_parts)
    
    # Update BENCHMARKS.md
    if not run_command(benchmarks_cmd, "Updating BENCHMARKS.md"):
        sys.exit(1)
    
    # Update README.md
    readme_cmd = "python update_readme_section.py"
    if not run_command(readme_cmd, "Updating README.md"):
        sys.exit(1)
    
    print("\n✅ Successfully updated both BENCHMARKS.md and README.md!")
    print("\n💡 To commit and push changes, run:")
    print("   git add BENCHMARKS.md README.md")
    print("   git commit -m 'Update benchmark tables'")
    print("   git push")

if __name__ == "__main__":
    main()