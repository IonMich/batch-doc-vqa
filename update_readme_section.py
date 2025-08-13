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
                'python', 'generate_benchmark_table.py', 
                '--readme', '--format', 'markdown', '--no-interactive',
                '--output', tmp_path
            ], capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Error generating table: {result.stderr}")
                return False
            
            with open(tmp_path, 'r', encoding='utf-8') as f:
                new_content = f.read().strip()
                
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    
    return update_readme_section(args.readme, new_content, args.start_marker, args.end_marker)

if __name__ == "__main__":
    main()