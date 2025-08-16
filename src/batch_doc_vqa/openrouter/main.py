#!/usr/bin/env python3
"""
Main entry point for OpenRouter inference.
"""

def main():
    """Main entry point that delegates to CLI."""
    from .cli import main as cli_main
    cli_main()

if __name__ == "__main__":
    main()