#!/usr/bin/env python3
"""
Generate Pareto frontier plot for 8-digit_top1 vs total cost.
"""
import matplotlib.pyplot as plt
import argparse
from typing import Dict, List, Tuple
from adjustText import adjust_text

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
                      title: str = "Model Performance vs Cost Trade-off (quiz-identify-vqa)", show_all_labels: bool = True):
    """Create Pareto frontier plot for 8-digit_top1 vs total cost."""
    
    # Extract data
    model_names = []
    orgs = []
    id_top1_scores = []
    total_costs = []
    cohort_sizes = []
    id_top1_ci_lows = []
    id_top1_ci_highs = []
    total_cost_ci_lows = []
    total_cost_ci_highs = []
    
    for model_key, data in run_stats.items():
        if not isinstance(data, dict):
            print(f"  Skipping {model_key}: invalid run data")
            continue

        run_info = data.get("run_info")
        stats = data.get("stats")
        if not isinstance(run_info, dict) or not isinstance(stats, dict):
            print(f"  Skipping {model_key}: missing run info or stats")
            continue

        config = run_info.get("config")
        if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
            print(f"  Skipping {model_key}: missing model configuration")
            continue

        try:
            total_cost = float(stats.get("total_cost", 0) or 0)
            id_top1 = float(stats.get("id_top1", 0) or 0)
        except (TypeError, ValueError):
            print(f"  Skipping {model_key}: non-numeric cost or score")
            continue

        # Skip models with zero cost (likely free or estimation failed)
        if total_cost <= 0:
            continue

        model_config = config["model"]
        org = model_config.get("org", "other")
        model_name = model_config.get("model", model_key)
        if model_config.get("variant"):
            model_name += f"-{model_config['variant']}"

        model_names.append(model_name)
        orgs.append(org)
        id_top1_scores.append(id_top1)
        total_costs.append(total_cost)
        n_runs = int(stats.get("n_runs", 1) or 1)
        cohort_sizes.append(n_runs)

        ci = stats.get("ci", {})
        id_ci = ci.get("id_top1") if isinstance(ci, dict) else None
        cost_ci = ci.get("total_cost") if isinstance(ci, dict) else None

        if (
            n_runs >= 3
            and isinstance(id_ci, list)
            and len(id_ci) == 2
            and isinstance(cost_ci, list)
            and len(cost_ci) == 2
        ):
            id_top1_ci_lows.append(float(id_ci[0]))
            id_top1_ci_highs.append(float(id_ci[1]))
            total_cost_ci_lows.append(float(cost_ci[0]))
            total_cost_ci_highs.append(float(cost_ci[1]))
        else:
            id_top1_ci_lows.append(None)
            id_top1_ci_highs.append(None)
            total_cost_ci_lows.append(None)
            total_cost_ci_highs.append(None)
    
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
        if (
            id_top1_ci_lows[i] is not None
            and id_top1_ci_highs[i] is not None
            and total_cost_ci_lows[i] is not None
            and total_cost_ci_highs[i] is not None
        ):
            xerr = [[cost - float(total_cost_ci_lows[i])], [float(total_cost_ci_highs[i]) - cost]]
            yerr = [[score - float(id_top1_ci_lows[i])], [float(id_top1_ci_highs[i]) - score]]
            ax.errorbar(
                cost,
                score,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=color,
                alpha=0.35,
                elinewidth=1,
                capsize=2,
                zorder=1,
            )
        
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
    
    # Collect text annotations for adjustText
    texts = []
    
    # Add labels for frontier models (black text)
    for i in frontier_indices:
        label = model_names[i]
        if cohort_sizes[i] > 1:
            label = f"{label} (n={cohort_sizes[i]})"
        text = ax.annotate(label, 
                          (total_costs[i], id_top1_scores[i]),
                          fontsize=9, ha='center', va='center',
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='black'),
                          zorder=4)
        texts.append(text)
    
    # Add labels for non-frontier models (gray text) if show_all_labels is True
    if show_all_labels:
        for i, (cost, score) in enumerate(zip(total_costs, id_top1_scores)):
            if i not in frontier_indices:
                label = model_names[i]
                if cohort_sizes[i] > 1:
                    label = f"{label} (n={cohort_sizes[i]})"
                text = ax.annotate(label, 
                                  (cost, score),
                                  fontsize=8, ha='center', va='center', color='gray',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='gray'),
                                  zorder=3)
                texts.append(text)
    
    # Formatting
    ax.set_xscale('log')
    ax.set_xlabel('Total Cost ($)', fontsize=12)
    ax.set_ylabel('8-digit ID Top-1 Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.text(
        0.01,
        0.99,
        "Test: quiz-identify-vqa",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        color="#444444",
    )
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
    
    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    
    # Apply adjustText LAST, after all plotting is complete
    # This is crucial according to the documentation
    if texts:
        adjust_text(texts,
                   ax=ax,
                   # Slightly increased expansion for better text separation
                   expand=(1.1, 1.25),
                   # Prevent axes from expanding beyond reasonable limits
                   ensure_inside_axes=True,
                   expand_axes=False,
                   # Limit horizontal movement, allow more vertical movement
                   max_move=(30, 100),
                   # Limit iterations to prevent runaway adjustments
                   iter_lim=150,
                   # Bias forces towards vertical movement (x_force, y_force)
                   force_text=(0.05, 0.6),  # Low horizontal, high vertical repulsion
                   force_explode=(0.05, 0.8),  # Initial vertical separation preferred
                   force_static=(0.05, 0.3),  # Reduced horizontal static force
                   force_pull=(0.01, 0.01),  # Keep pull forces balanced
                   # Add arrows to connect moved labels to points
                   arrowprops=dict(arrowstyle='-', color='gray', alpha=0.6, lw=0.5))
    
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
    parser.add_argument("--title", default="Model Performance vs Cost Trade-off (quiz-identify-vqa)", help="Plot title")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive model review")
    parser.add_argument("--hide-non-frontier-labels", action="store_true", 
                       help="Hide labels for non-frontier models (default: show all labels in gray)")
    parser.add_argument("--cohort-window-hours", type=float, default=24.0,
                       help="Window (hours) for grouping latest cohorts by matching prompt hash + git commit")
    parser.add_argument("--debug-cohorts", action="store_true",
                       help="Print latest-cohort grouping details before stats generation")
    args = parser.parse_args()
    
    # Generate benchmark data
    generator = BenchmarkTableGenerator(args.runs_dir, interactive=not args.no_interactive)
    
    run_stats = generator.build_run_stats(
        run_patterns=args.patterns,
        doc_info_file=args.doc_info,
        test_ids_file=args.test_ids,
        cohort_window_hours=args.cohort_window_hours,
        debug_cohorts=args.debug_cohorts,
    )
    if not run_stats:
        print("No runs found matching the criteria")
        return
    
    # Create plot
    create_pareto_plot(run_stats, args.output, args.title, show_all_labels=not args.hide_non_frontier_labels)


if __name__ == "__main__":
    main()
