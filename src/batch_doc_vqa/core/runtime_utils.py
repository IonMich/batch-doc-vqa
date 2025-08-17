#!/usr/bin/env python3
"""
Shared runtime utilities for batch document VQA.
"""


def format_runtime(seconds: float) -> str:
    """Format runtime in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f} minutes"
    else:
        hours = seconds / 3600
        return f"{hours:.1f} hours"