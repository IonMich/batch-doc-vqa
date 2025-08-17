#!/usr/bin/env python3
"""
Shared progress tracking utilities for batch document VQA.
"""
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn


def create_inference_progress() -> Progress:
    """Create a Rich progress instance with standard columns for inference tracking."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TextColumn("[bold green]{task.fields[success_rate]}"),
        TextColumn("•"),
        TextColumn("{task.fields[last_result]}", style="dim")
    )


def add_inference_task(progress: Progress, total_images: int):
    """Add an inference task to the progress tracker with standard fields."""
    return progress.add_task(
        "Processing images...", 
        total=total_images,
        success_rate="0%",
        last_result=""
    )