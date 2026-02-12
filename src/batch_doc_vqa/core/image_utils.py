#!/usr/bin/env python3
"""
Shared image processing utilities for batch document VQA.
"""
import csv
import os
import re
import base64
from pathlib import Path
from typing import Optional, Sequence


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


def get_imagepaths_from_doc_info(
    doc_info_file: str,
    *,
    images_dir: Optional[str] = None,
    pages: Optional[Sequence[int]] = None,
) -> list[str]:
    """Load image paths from a doc_info CSV (doc,page,filename)."""
    doc_info_path = Path(doc_info_file)
    base_dir = Path(images_dir) if images_dir is not None else doc_info_path.parent
    page_filter = set(int(page) for page in pages) if pages else None

    images: list[str] = []
    with open(doc_info_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename_raw = str((row or {}).get("filename", "")).strip()
            if not filename_raw:
                continue

            if page_filter is not None:
                page_text = str((row or {}).get("page", "")).strip()
                if not page_text:
                    continue
                try:
                    page_val = int(page_text)
                except ValueError:
                    continue
                if page_val not in page_filter:
                    continue

            filename_path = Path(filename_raw)
            if filename_path.is_absolute():
                images.append(str(filename_path))
            else:
                images.append(str(base_dir / filename_path))

    # Remove duplicates while preserving deterministic natural sort.
    unique_images = sorted(set(images), key=natural_sort_key)
    return unique_images


def natural_sort_key(s: str) -> list:
    """Natural sorting key for filenames with numbers."""
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]
