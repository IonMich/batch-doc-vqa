#!/usr/bin/env python3
"""
Update README.md benchmark table section using template markers.
"""
import re
import argparse
from pathlib import Path

def update_readme_section(readme_path: str, new_content: str, 
                         start_marker: str = "<!-- BENCHMARK_TABLE_START -->",
                         end_marker: str = "<!-- BENCHMARK_TABLE_END -->") -> bool:
    """Update section in README.md between markers."""
    
    readme_file = Path(readme_path)
    if not readme_file.exists():
        print(f"Error: {readme_path} not found")
        return False
    
    # Read current README
    with open(readme_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Check if markers exist
    if start_marker not in content or end_marker not in content:
        print(f"Error: Markers not found in {readme_path}")
        print(f"Please add '{start_marker}' and '{end_marker}' to your README.md")
        return False
    
    # Replace content between markers
    pattern = f"({re.escape(start_marker)}).*?({re.escape(end_marker)})"
    replacement = f"{start_marker}\\n\\n{new_content}\\n\\n{end_marker}"
    
    new_readme_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    # Write updated README
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write(new_readme_content)
    
    print(f"✅ Updated benchmark section in {readme_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description="Update README.md benchmark section")
    parser.add_argument("--readme", default="README.md", help="Path to README.md file")
    parser.add_argument("--table-file", help="File containing new table content")
    parser.add_argument("--start-marker", default="<!-- BENCHMARK_TABLE_START -->", 
                       help="Start marker for section")
    parser.add_argument("--end-marker", default="<!-- BENCHMARK_TABLE_END -->", 
                       help="End marker for section")
    args = parser.parse_args()
    
    if args.table_file:
        # Read new content from file
        with open(args.table_file, 'r', encoding='utf-8') as f:
            new_content = f.read().strip()
    else:
        # Generate table content using benchmark generator
        import subprocess
        import tempfile
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as tmp_f:
            tmp_path = tmp_f.name
        
        try:
            # Generate README table
            result = subprocess.run([
                'uv', 'run', 'generate-benchmark-table', 
                '--readme', '--format', 'markdown', '--no-interactive',
                '--output', tmp_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error generating table: {result.stderr}")
                print(f"Command stdout: {result.stdout}")
                return False
            
            with open(tmp_path, 'r', encoding='utf-8') as f:
                table_content = f.read().strip()
            
            # Add Pareto plot if it exists
            pareto_plot_path = "pareto_plot.png"
            if Path(pareto_plot_path).exists():
                new_content = f"{table_content}\n\n### Performance vs Cost Trade-off\n\nThe chart below shows the Pareto frontier of models, highlighting the most cost-efficient options for different performance levels:\n\n![Model Performance vs Cost Trade-off](pareto_plot.png)"
            else:
                new_content = table_content
                
        except Exception as e:
            print(f"Exception during table generation: {e}")
            return False
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    return update_readme_section(args.readme, new_content, args.start_marker, args.end_marker)

def cli_main():
    """Entry point for the CLI command."""
    import sys
    try:
        success = main()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"Exception in main(): {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    cli_main()