#!/usr/bin/env python3
"""
Generate Pareto frontier plot for 8-digit_top1 vs total cost.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import json

from .table_generator import BenchmarkTableGenerator


def calculate_pareto_frontier(points: List[Tuple[float, float]]) -> List[int]:
    """
    Calculate Pareto frontier indices for maximizing y (accuracy) and minimizing x (cost).
    Returns indices of points on the frontier.
    """
    # Sort by x (cost) ascending
    sorted_indices = sorted(range(len(points)), key=lambda i: points[i][0])
    
    frontier_indices = []
    max_y_so_far = -float('inf')
    
    for idx in sorted_indices:
        x, y = points[idx]
        if y > max_y_so_far:
            frontier_indices.append(idx)
            max_y_so_far = y
    
    return frontier_indices


def get_organization_colors():
    """Get consistent colors for organizations."""
    return {
        'openai': '#00A67E',      # OpenAI green
        'anthropic': '#D97757',   # Anthropic orange
        'google': '#4285F4',      # Google blue
        'meta-llama': '#0866FF',  # Meta blue
        'qwen': '#FF6B35',        # Qwen orange
        'z-ai': '#9C27B0',        # Purple
        'moonshotai': '#FF9800',  # Orange
        'mistralai': '#FF5722',   # Red-orange
        'microsoft': '#00BCF2',   # Microsoft blue
        'other': '#666666'        # Gray for others
    }


def create_pareto_plot(run_stats: Dict, output_path: str = "pareto_plot.png", 
                      title: str = "Model Performance vs Cost Trade-off"):
    """Create Pareto frontier plot for 8-digit_top1 vs total cost."""
    
    # Extract data
    model_names = []
    orgs = []
    id_top1_scores = []
    total_costs = []
    
    for model_key, data in run_stats.items():
        config = data["run_info"]["config"]
        stats = data["stats"]
        
        # Skip models with zero cost (likely free or estimation failed)
        if stats.get('total_cost', 0) <= 0:
            continue
            
        org = config['model']['org']
        model_name = config['model']['model']
        if config['model']['variant']:
            model_name += f"-{config['model']['variant']}"
        
        model_names.append(model_name)
        orgs.append(org)
        id_top1_scores.append(stats['id_top1'])
        total_costs.append(stats['total_cost'])
    
    if len(model_names) == 0:
        print("No models with cost data found")
        return
    
    # Calculate Pareto frontier
    points = list(zip(total_costs, id_top1_scores))
    frontier_indices = calculate_pareto_frontier(points)
    frontier_indices.sort(key=lambda i: total_costs[i])  # Sort by cost for line plotting
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color scheme
    org_colors = get_organization_colors()
    
    # Plot all points
    for i, (org, cost, score) in enumerate(zip(orgs, total_costs, id_top1_scores)):
        color = org_colors.get(org, org_colors['other'])
        
        if i in frontier_indices:
            # Frontier points: full opacity, larger size
            ax.scatter(cost, score, c=color, s=100, alpha=1.0, edgecolors='black', linewidth=1, zorder=3)
        else:
            # Non-frontier points: faded, smaller size  
            ax.scatter(cost, score, c=color, s=60, alpha=0.4, edgecolors='gray', linewidth=0.5, zorder=2)
    
    # Draw Pareto frontier line
    if len(frontier_indices) > 1:
        frontier_costs = [total_costs[i] for i in frontier_indices]
        frontier_scores = [id_top1_scores[i] for i in frontier_indices]
        ax.plot(frontier_costs, frontier_scores, 'k--', alpha=0.7, linewidth=2, zorder=1, label='Pareto Frontier')
    
    # Add labels only for frontier models
    for i in frontier_indices:
        ax.annotate(model_names[i], 
                   (total_costs[i], id_top1_scores[i]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=9, ha='left', va='bottom',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                   zorder=4)
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Total Cost ($)', fontsize=12)
    ax.set_ylabel('8-digit ID Top-1 Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Create legend for organizations
    legend_elements = []
    used_orgs = set(orgs)
    for org in sorted(used_orgs):
        color = org_colors.get(org, org_colors['other'])
        legend_elements.append(plt.scatter([], [], c=color, s=80, label=org, alpha=0.8))
    
    # Add Pareto frontier to legend if it exists
    if len(frontier_indices) > 1:
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', 
                                        linewidth=2, label='Pareto Frontier'))
    
    ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Pareto plot saved to {output_path}")
    
    # Print frontier models
    print(f"\nModels on Pareto frontier ({len(frontier_indices)}):")
    for i in frontier_indices:
        print(f"  {model_names[i]} ({orgs[i]}): {id_top1_scores[i]:.1f}% accuracy, ${total_costs[i]:.4f} total cost")
    
    return frontier_indices


def main():
    parser = argparse.ArgumentParser(description="Generate Pareto frontier plot")
    parser.add_argument("--runs-dir", default="tests/output/runs", help="Base directory for runs")
    parser.add_argument("--patterns", nargs="*", help="Patterns to filter runs")
    parser.add_argument("--doc-info", default="imgs/q11/doc_info.csv", help="Document info CSV")
    parser.add_argument("--test-ids", default="tests/data/test_ids.csv", help="Test IDs CSV") 
    parser.add_argument("--output", default="pareto_plot.png", help="Output plot file")
    parser.add_argument("--title", default="Model Performance vs Cost Trade-off", help="Plot title")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive model review")
    args = parser.parse_args()
    
    # Generate benchmark data
    generator = BenchmarkTableGenerator(args.runs_dir, interactive=not args.no_interactive)
    
    # Get all matching runs
    all_runs = []
    if args.patterns:
        for pattern in args.patterns:
            runs = generator.run_manager.list_runs(pattern)
            all_runs.extend(runs)
    else:
        all_runs = generator.run_manager.list_runs()
    
    if not all_runs:
        print("No runs found matching the criteria")
        return
    
    # Remove duplicates and get latest run per model
    runs_by_model = {}
    unknown_models = []
    for run_info in all_runs:
        config = run_info["config"]
        model_key = generator._get_model_key(config)
        
        # Check if model metadata exists
        if model_key not in generator.model_metadata["models"]:
            if model_key not in unknown_models:
                unknown_models.append(model_key)
        
        # Keep the latest run for each model
        if model_key not in runs_by_model or run_info["config"]["run_info"]["timestamp"] > runs_by_model[model_key]["config"]["run_info"]["timestamp"]:
            runs_by_model[model_key] = run_info
    
    # Review unknown models
    generator._review_unknown_models(unknown_models)
    
    # Compute stats for each run
    run_stats = {}
    for model_key, run_info in runs_by_model.items():
        print(f"Processing run: {run_info['run_name']}")
        stats = generator.compute_run_stats(run_info, args.doc_info, args.test_ids)
        run_stats[model_key] = {
            "run_info": run_info,
            "stats": stats
        }
    
    # Create plot
    create_pareto_plot(run_stats, args.output, args.title)


if __name__ == "__main__":
    main()