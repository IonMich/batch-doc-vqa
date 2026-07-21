#!/usr/bin/env python3
"""
Generate Pareto frontier plot(s) for benchmark metric(s) vs total cost.
"""
import matplotlib.pyplot as plt
import argparse
import textwrap
from typing import Dict, List, Tuple
from adjustText import adjust_text

from .table_generator import BenchmarkTableGenerator
from .published_runs import DEFAULT_PUBLISHED_RUNS_DIR


def calculate_pareto_frontier(points: List[Tuple[float, float]], maximize_y: bool = True) -> List[int]:
    """
    Calculate Pareto frontier indices for minimizing x (cost) and optimizing y.
    Returns indices of points on the frontier.
    """
    # Sort by x (cost) ascending
    sorted_indices = sorted(range(len(points)), key=lambda i: points[i][0])

    frontier_indices = []
    best_y_so_far = -float("inf") if maximize_y else float("inf")

    for idx in sorted_indices:
        x, y = points[idx]
        if (maximize_y and y > best_y_so_far) or (not maximize_y and y < best_y_so_far):
            frontier_indices.append(idx)
            best_y_so_far = y

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


def _reasoning_label(config: Dict) -> str:
    additional = config.get("additional", {}) if isinstance(config, dict) else {}
    generation_params = (
        additional.get("generation_params_effective", {})
        if isinstance(additional, dict)
        else {}
    )
    reasoning = generation_params.get("reasoning") if isinstance(generation_params, dict) else None
    if isinstance(reasoning, dict):
        effort = reasoning.get("effort")
        if isinstance(effort, str) and effort.strip():
            return effort.strip()
        enabled = reasoning.get("enabled")
        if enabled is False:
            return "none"
        if enabled is True:
            return "enabled"
    elif isinstance(reasoning, str) and reasoning.strip():
        return reasoning.strip()
    return "default"


def _plot_model_label(config: Dict, fallback: str) -> str:
    model_config = config.get("model", {}) if isinstance(config, dict) else {}
    model_name = model_config.get("model", fallback)
    if model_config.get("variant"):
        model_name += f"-{model_config['variant']}"
    reasoning = _reasoning_label(config)
    if reasoning != "default":
        model_name += f" ({reasoning})"
    return model_name



def create_pareto_plot(
    run_stats: Dict,
    output_path: str = "pareto_plot.png",
    title: str = "Model Performance vs Cost Trade-off (quiz-identify-vqa)",
    *,
    label_mode: str = "frontier",
    y_metric: str = "id_top1",
    y_axis_label: str = "8-digit ID Top-1 Accuracy (%)",
    y_metric_print_label: str = "accuracy",
    y_metric_print_decimals: int = 1,
    y_metric_suffix: str = "%",
    maximize_y: bool = True,
    invert_y_axis: bool = False,
):
    """Create Pareto frontier plot for a chosen y-axis metric vs total cost."""

    if label_mode not in {"all", "frontier", "none"}:
        raise ValueError(f"Unsupported label mode: {label_mode}")

    # Extract data
    model_names = []
    orgs = []
    y_values = []
    total_costs = []
    cohort_sizes = []
    y_ci_lows = []
    y_ci_highs = []
    total_cost_ci_lows = []
    total_cost_ci_highs = []
    excluded_cost_statuses: Dict[str, int] = {}

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
            total_cost_value = stats.get("total_cost")
            y_value = float(stats.get(y_metric, 0) or 0)
        except (TypeError, ValueError):
            print(f"  Skipping {model_key}: non-numeric cost or score")
            continue

        # Direct callers from older integrations may supply only a numeric
        # total. Cached repository runs are always recalculated to v2 first.
        cost_status = str(stats.get("cost_status") or "precise")
        if cost_status not in {"precise", "estimated", "verified-zero"}:
            excluded_cost_statuses[cost_status] = excluded_cost_statuses.get(cost_status, 0) + 1
            continue
        if not isinstance(total_cost_value, (int, float)):
            excluded_cost_statuses[cost_status] = excluded_cost_statuses.get(cost_status, 0) + 1
            continue
        total_cost = float(total_cost_value)

        # A verified zero is valid provenance, but cannot be placed on the log
        # axis.  It is reported explicitly rather than misrepresented.
        if total_cost <= 0:
            excluded_cost_statuses[cost_status] = excluded_cost_statuses.get(cost_status, 0) + 1
            continue

        model_config = config["model"]
        org = model_config.get("org", "other")
        model_name = _plot_model_label(config, model_key)

        model_names.append(model_name)
        orgs.append(org)
        y_values.append(y_value)
        total_costs.append(total_cost)
        n_runs = int(stats.get("n_runs", 1) or 1)
        cohort_sizes.append(n_runs)

        ci = stats.get("ci", {})
        y_ci = ci.get(y_metric) if isinstance(ci, dict) else None
        cost_ci = ci.get("total_cost") if isinstance(ci, dict) else None

        if (
            n_runs >= 3
            and isinstance(y_ci, list)
            and len(y_ci) == 2
            and isinstance(cost_ci, list)
            and len(cost_ci) == 2
        ):
            y_ci_lows.append(float(y_ci[0]))
            y_ci_highs.append(float(y_ci[1]))
            total_cost_ci_lows.append(float(cost_ci[0]))
            total_cost_ci_highs.append(float(cost_ci[1]))
        else:
            y_ci_lows.append(None)
            y_ci_highs.append(None)
            total_cost_ci_lows.append(None)
            total_cost_ci_highs.append(None)

    if len(model_names) == 0:
        print("No models with cost data found")
        return

    if excluded_cost_statuses:
        status_text = ", ".join(
            f"{status}: {count}" for status, count in sorted(excluded_cost_statuses.items())
        )
        print(f"Excluded from log-cost plot by cost provenance: {status_text}")

    # Calculate Pareto frontier
    points = list(zip(total_costs, y_values))
    frontier_indices = calculate_pareto_frontier(points, maximize_y=maximize_y)
    frontier_indices.sort(key=lambda i: total_costs[i])  # Sort by cost for line plotting

    # Keep the chart itself unchanged.  Frontier labels are placed immediately
    # adjacent to their points, using a fixed, alternating layout rather than
    # a global repulsion pass that can send labels far across the plot.
    fig, ax = plt.subplots(figsize=(12, 8))

    # Color scheme
    org_colors = get_organization_colors()

    # Plot all points
    for i, (org, cost, score) in enumerate(zip(orgs, total_costs, y_values)):
        color = org_colors.get(org, org_colors["other"])
        if (
            y_ci_lows[i] is not None
            and y_ci_highs[i] is not None
            and total_cost_ci_lows[i] is not None
            and total_cost_ci_highs[i] is not None
        ):
            xerr = [[cost - float(total_cost_ci_lows[i])], [float(total_cost_ci_highs[i]) - cost]]
            yerr = [[score - float(y_ci_lows[i])], [float(y_ci_highs[i]) - score]]
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
        frontier_scores = [y_values[i] for i in frontier_indices]
        ax.plot(frontier_costs, frontier_scores, 'k--', alpha=0.7, linewidth=2, zorder=1, label='Pareto Frontier')

    # Labeling every model makes the dense upper-right cluster unreadable.
    # For the small frontier set, use deliberately chosen nearby anchors.  The
    # only long name is wrapped to prevent it from colliding with its neighbour.
    texts = []

    if label_mode == "frontier":
        # Ordered from lowest to highest cost.  The fifth and sixth points are
        # the only close pair in the current frontier, so their labels point in
        # opposite directions.  The remaining positions keep labels beside,
        # rather than away from, their data points.
        frontier_label_positions = (
            (8, 8, "left", "bottom"),
            (8, 8, "left", "bottom"),
            (8, 8, "left", "bottom"),
            (8, 8, "left", "bottom"),
            (-8, -8, "right", "top"),
            (8, 8, "left", "bottom"),
            (8, -8, "left", "top"),
        )
        for position, i in enumerate(frontier_indices):
            dx, dy, horizontal_alignment, vertical_alignment = frontier_label_positions[
                min(position, len(frontier_label_positions) - 1)
            ]
            label = textwrap.fill(
                model_names[i], width=25, break_long_words=False, break_on_hyphens=True
            )
            if cohort_sizes[i] > 1:
                label = f"{label} (n={cohort_sizes[i]})"
            ax.annotate(
                label,
                (total_costs[i], y_values[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=8,
                ha=horizontal_alignment,
                va=vertical_alignment,
                bbox=dict(boxstyle="round,pad=0.16", facecolor="white", alpha=0.82, edgecolor="none"),
                zorder=4,
            )

    elif label_mode == "all":
        # Keep the legacy exhaustive mode for investigations and small cohorts.
        for i in frontier_indices:
            label = model_names[i]
            if cohort_sizes[i] > 1:
                label = f"{label} (n={cohort_sizes[i]})"
            text = ax.annotate(
                label,
                (total_costs[i], y_values[i]),
                fontsize=9,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="black"),
                zorder=4,
            )
            texts.append(text)

    if label_mode == "all":
        for i, (cost, score) in enumerate(zip(total_costs, y_values)):
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
    ax.set_ylabel(y_axis_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    if invert_y_axis:
        ax.invert_yaxis()
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
        color = org_colors.get(org, org_colors["other"])
        legend_elements.append(plt.scatter([], [], c=color, s=80, label=org, alpha=0.8))

    # Add Pareto frontier to legend if it exists
    if len(frontier_indices) > 1:
        legend_elements.append(plt.Line2D([0], [0], color='black', linestyle='--', 
                                        linewidth=2, label='Pareto Frontier'))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # The exhaustive mode retains the collision solver for small,
    # investigative cohorts.  Frontier labels use the compact fixed layout
    # above, so they remain near their points.
    if texts and label_mode == "all":
        adjust_text(
            texts,
            ax=ax,
            expand=(1.1, 1.25),
            ensure_inside_axes=True,
            expand_axes=False,
            max_move=(30, 100),
            iter_lim=150,
            force_text=(0.05, 0.6),
            force_explode=(0.05, 0.8),
            force_static=(0.05, 0.3),
            force_pull=(0.01, 0.01),
            arrowprops=dict(arrowstyle='-', color='gray', alpha=0.6, lw=0.5),
        )

    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Pareto plot saved to {output_path}")
    
    # Print frontier models
    print(f"\nModels on Pareto frontier ({len(frontier_indices)}):")
    for i in frontier_indices:
        metric_text = f"{y_values[i]:.{y_metric_print_decimals}f}{y_metric_suffix} {y_metric_print_label}"
        print(f"  {model_names[i]} ({orgs[i]}): {metric_text}, ${total_costs[i]:.4f} total cost")

    return frontier_indices


def main():
    parser = argparse.ArgumentParser(description="Generate Pareto frontier plot")
    parser.add_argument("--runs-dir", default="tests/output/runs", help="Base directory for runs")
    parser.add_argument("--source", choices=("auto", "local", "published"), default="auto",
                       help="Use the finalized published archive when available (default: auto).")
    parser.add_argument("--published-runs-dir", default=str(DEFAULT_PUBLISHED_RUNS_DIR),
                       help="Directory containing sanitized published run summaries.")
    parser.add_argument("--patterns", nargs="*", help="Patterns to filter runs")
    parser.add_argument("--doc-info", default="imgs/q11/doc_info.csv", help="Document info CSV")
    parser.add_argument("--test-ids", default="tests/data/test_ids.csv", help="Test IDs CSV") 
    parser.add_argument("--output", default="pareto_plot.png", help="Output plot file")
    parser.add_argument("--title", default="Model Performance vs Cost Trade-off (quiz-identify-vqa)", help="Plot title")
    parser.add_argument(
        "--extra-id-lev-pareto",
        action="store_true",
        help="Generate an additional Pareto plot using ID Avg d_Lev vs cost (non-default)",
    )
    parser.add_argument(
        "--id-lev-output",
        default="pareto_plot_id_lev.png",
        help="Output file for optional ID Avg d_Lev Pareto plot",
    )
    parser.add_argument(
        "--id-lev-title",
        default="Model ID Levenshtein Distance vs Cost (quiz-identify-vqa)",
        help="Title for optional ID Avg d_Lev Pareto plot",
    )
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive model review")
    parser.add_argument("--no-cache", action="store_true", help="Do not write machine-local table_results caches")
    parser.add_argument(
        "--label-mode",
        choices=("frontier", "none", "all"),
        default="frontier",
        help="Which model names to print: frontier (default), none, or all.",
    )
    parser.add_argument(
        "--hide-non-frontier-labels",
        action="store_true",
        help="Deprecated alias for --label-mode frontier.",
    )
    parser.add_argument("--cohort-window-hours", type=float, default=24.0,
                       help="Window (hours) for grouping latest cohorts by matching prompt hash + git commit")
    parser.add_argument("--debug-cohorts", action="store_true",
                       help="Print latest-cohort grouping details before stats generation")
    args = parser.parse_args()
    
    # Generate benchmark data
    generator = BenchmarkTableGenerator(
        args.runs_dir,
        interactive=not args.no_interactive,
        source=args.source,
        published_runs_dir=args.published_runs_dir,
        cache_results=not args.no_cache,
        write_metadata=not args.no_cache,
    )
    
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
    
    label_mode = "frontier" if args.hide_non_frontier_labels else args.label_mode

    # Create plot
    create_pareto_plot(run_stats, args.output, args.title, label_mode=label_mode)
    if args.extra_id_lev_pareto:
        create_pareto_plot(
            run_stats,
            args.id_lev_output,
            args.id_lev_title,
            label_mode=label_mode,
            y_metric="id_avg_lev",
            y_axis_label="ID Avg d_Lev (lower is better)",
            y_metric_print_label="avg ID d_Lev",
            y_metric_print_decimals=4,
            y_metric_suffix="",
            maximize_y=False,
            invert_y_axis=True,
        )


if __name__ == "__main__":
    main()
