#!/usr/bin/env python3
"""
Shared image processing utilities for batch document VQA.
"""
import os
import re
import base64


def filepath_to_base64(filepath: str) -> str:
    """Convert file to base64 encoded string."""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_imagepaths(folder: str, pattern: str) -> list[str]:
    """Get image paths matching pattern from folder."""
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if re.match(pattern, file):
                images.append(os.path.join(root, file))
    # sort by integers in the filename
    images.sort(key=natural_sort_key)
    return images


def natural_sort_key(s: str) -> list:
    """Natural sorting key for filenames with numbers."""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]