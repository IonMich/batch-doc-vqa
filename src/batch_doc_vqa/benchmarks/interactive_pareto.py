#!/usr/bin/env python3
"""Generate a standalone, filterable benchmark Pareto explorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .table_generator import BenchmarkTableGenerator


def _extract_points(run_stats: Dict[str, Any], y_metric: str) -> List[Dict[str, Any]]:
    """Return the cost/metric rows shared by the static and interactive plots."""

    points: List[Dict[str, Any]] = []
    for model_key, data in run_stats.items():
        if not isinstance(data, dict):
            continue
        run_info = data.get("run_info")
        stats = data.get("stats")
        if not isinstance(run_info, dict) or not isinstance(stats, dict):
            continue
        config = run_info.get("config")
        model_config = config.get("model") if isinstance(config, dict) else None
        if not isinstance(model_config, dict):
            continue
        try:
            total_cost = float(stats.get("total_cost", 0) or 0)
            accuracy = float(stats.get(y_metric, 0) or 0)
        except (TypeError, ValueError):
            continue
        if total_cost <= 0:
            # A log cost scale cannot faithfully place a zero-cost point.  This
            # matches the established static Pareto plot behaviour.
            continue

        model_name = str(model_config.get("model") or model_key)
        variant = model_config.get("variant")
        if variant:
            model_name = f"{model_name}-{variant}"
        points.append(
            {
                "organization": str(model_config.get("org") or "other"),
                "model": model_name,
                "cost": total_cost,
                "accuracy": accuracy,
            }
        )
    return sorted(points, key=lambda item: (item["cost"], item["organization"], item["model"]))


def calculate_pareto_frontier(points: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return the lowest-cost model for each strictly improved accuracy level."""

    frontier: List[Dict[str, Any]] = []
    best_accuracy = float("-inf")
    for point in sorted(points, key=lambda item: (item["cost"], -item["accuracy"])):
        if point["accuracy"] > best_accuracy:
            frontier.append(point)
            best_accuracy = point["accuracy"]
    return frontier


_HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>__TITLE__</title>
  <style>
    :root {
      color-scheme: light dark;
      --background: Canvas;
      --foreground: CanvasText;
      --muted: GrayText;
      --grid: color-mix(in srgb, CanvasText 20%, Canvas);
      --border: color-mix(in srgb, CanvasText 30%, Canvas);
      --surface: color-mix(in srgb, Canvas 92%, CanvasText);
      --frontier: Highlight;
      --frontier-text: HighlightText;
      --org-lightness: 40%;
    }
    @media (prefers-color-scheme: dark) {
      :root { --org-lightness: 68%; }
    }
    * { box-sizing: border-box; }
    body { background: var(--background); color: var(--foreground); font-family: system-ui, sans-serif; margin: 0; }
    main { margin: 0 auto; max-width: 1120px; padding: 1.5rem; }
    h1 { font-size: 1.35rem; font-weight: 600; margin: 0 0 0.35rem; }
    .subtitle { color: var(--muted); margin: 0 0 1rem; }
    .controls { align-items: center; display: flex; flex-wrap: wrap; gap: 0.75rem 1rem; margin: 0 0 0.75rem; }
    .organizations { align-items: center; display: flex; flex-wrap: wrap; gap: 0.4rem 0.9rem; margin: 0 0 1rem; }
    .organizations strong { font-size: 0.9rem; }
    .organization { align-items: center; display: inline-flex; gap: 0.35rem; }
    .swatch { background: var(--org-color); border-radius: 999px; display: inline-block; height: 0.65rem; width: 0.65rem; }
    button, select { background: var(--background); border: 1px solid var(--border); border-radius: 0.35rem; color: var(--foreground); font: inherit; padding: 0.35rem 0.55rem; }
    button { cursor: pointer; }
    label { align-items: center; display: inline-flex; gap: 0.4rem; }
    .chart-wrap { position: relative; }
    svg { display: block; height: auto; width: 100%; }
    .axis, .grid { stroke: var(--grid); stroke-width: 1; }
    .axis { stroke: var(--border); }
    .tick { fill: var(--muted); font-size: 12px; }
    .axis-label, .frontier-label { fill: var(--foreground); font-size: 13px; }
    .frontier-label { font-size: 11px; font-weight: 600; }
    .frontier-line { fill: none; stroke: var(--frontier); stroke-width: 2; }
    .point { cursor: pointer; opacity: 0.58; stroke: var(--background); stroke-width: 1.5; }
    .point.frontier-point { opacity: 1; stroke: var(--foreground); stroke-width: 1.5; }
    .point.inspected { opacity: 1; stroke: var(--foreground); stroke-width: 3; }
    .tooltip { background: var(--surface); border: 1px solid var(--border); border-radius: 0.35rem; max-width: 20rem; padding: 0.45rem 0.6rem; pointer-events: none; position: absolute; visibility: hidden; }
    .status { color: var(--muted); min-height: 1.4em; }
    .inspectors { align-items: end; display: flex; flex-wrap: wrap; gap: 0.75rem; margin-top: 0.75rem; }
    .inspectors label { align-items: start; flex-direction: column; font-size: 0.9rem; }
    .detail { margin-top: 0.6rem; min-height: 1.4em; }
    @media (max-width: 540px) { main { padding: 1rem; } .inspectors label, .inspectors select { width: 100%; } }
  </style>
</head>
<body>
  <main>
    <h1>__TITLE__</h1>
    <p class="subtitle">Cost is logarithmic. Point color identifies the model organization.</p>
    <section class="controls" aria-label="Plot controls">
      <label><input type="checkbox" id="pareto-only"> Pareto frontier only</label>
      <label><input type="checkbox" id="frontier-labels"> Label frontier points</label>
      <button type="button" id="select-all">All organizations</button>
      <button type="button" id="clear-all">Clear organizations</button>
    </section>
    <section class="organizations" id="organizations" aria-label="Organizations"><strong>Organizations</strong></section>
    <p class="status" id="status" aria-live="polite"></p>
    <div class="chart-wrap" id="chart-wrap">
      <svg id="chart" viewBox="0 0 900 500" role="img" aria-labelledby="chart-title chart-description">
        <title id="chart-title">Cost versus ID accuracy</title>
        <desc id="chart-description">Interactive scatter plot of benchmark cost and ID accuracy.</desc>
      </svg>
      <div id="tooltip" class="tooltip" role="status"></div>
    </div>
    <section class="inspectors" aria-label="Inspect a model">
      <label for="inspect-org">Inspect organization<select id="inspect-org"><option value="">Choose an organization</option></select></label>
      <label for="inspect-model">Inspect model<select id="inspect-model" disabled><option value="">Choose an organization first</option></select></label>
    </section>
    <output id="detail" class="detail" aria-live="polite">Choose an organization and model, or hover a point, to inspect it.</output>
  </main>
  <script>
    const data = __DATA__;
    const chart = document.getElementById("chart");
    const chartWrap = document.getElementById("chart-wrap");
    const organizations = document.getElementById("organizations");
    const paretoOnly = document.getElementById("pareto-only");
    const frontierLabels = document.getElementById("frontier-labels");
    const status = document.getElementById("status");
    const tooltip = document.getElementById("tooltip");
    const inspectOrg = document.getElementById("inspect-org");
    const inspectModel = document.getElementById("inspect-model");
    const detail = document.getElementById("detail");
    const dimensions = { width: 900, height: 500, left: 78, right: 28, top: 24, bottom: 62 };
    const costDomain = { min: 0.001, max: Math.pow(10, Math.ceil(Math.log10(Math.max(...data.map((item) => item.cost))))) };
    const accuracyDomain = { min: Math.max(0, Math.floor(Math.min(...data.map((item) => item.accuracy)) / 10) * 10 - 5), max: 102 };
    const orgs = [...new Set(data.map((item) => item.organization))].sort();
    const selectedOrgs = new Set(orgs);
    const orgColors = new Map(orgs.map((organization, index) => [organization, `hsl(${(index * 137.508) % 360} 62% var(--org-lightness))`]));
    let inspectedId = null;
    const svg = (name, attributes = {}) => {
      const element = document.createElementNS("http://www.w3.org/2000/svg", name);
      Object.entries(attributes).forEach(([key, value]) => element.setAttribute(key, value));
      return element;
    };
    const x = (cost) => dimensions.left + ((Math.log10(cost) - Math.log10(costDomain.min)) / (Math.log10(costDomain.max) - Math.log10(costDomain.min))) * (dimensions.width - dimensions.left - dimensions.right);
    const y = (accuracy) => dimensions.top + (1 - (accuracy - accuracyDomain.min) / (accuracyDomain.max - accuracyDomain.min)) * (dimensions.height - dimensions.top - dimensions.bottom);
    const formatCost = (cost) => new Intl.NumberFormat("en-US", { style: "currency", currency: "USD", maximumFractionDigits: cost < 0.1 ? 3 : 2 }).format(cost);
    const frontier = (items) => {
      let best = -Infinity;
      return [...items].sort((a, b) => a.cost - b.cost || b.accuracy - a.accuracy).filter((item) => {
        if (item.accuracy > best) { best = item.accuracy; return true; }
        return false;
      });
    };
    const setDetail = (item) => {
      inspectedId = item ? `${item.organization}/${item.model}` : null;
      detail.textContent = item ? `${item.organization}/${item.model} — ${formatCost(item.cost)} total cost · ${item.accuracy.toFixed(2)}% ID accuracy` : "Choose an organization and model, or hover a point, to inspect it.";
    };
    const syncModelSelector = () => {
      const organization = inspectOrg.value;
      inspectModel.replaceChildren();
      if (!organization) {
        inspectModel.disabled = true;
        inspectModel.append(new Option("Choose an organization first", ""));
        return;
      }
      inspectModel.disabled = false;
      inspectModel.append(new Option("Choose a model", ""));
      data.filter((item) => item.organization === organization).sort((a, b) => a.model.localeCompare(b.model)).forEach((item) => inspectModel.append(new Option(item.model, item.model)));
    };
    const selectForInspection = (item) => {
      inspectOrg.value = item.organization;
      syncModelSelector();
      inspectModel.value = item.model;
      setDetail(item);
    };
    const hideTooltip = () => { tooltip.style.visibility = "hidden"; };
    const showTooltip = (event, item) => {
      tooltip.textContent = `${item.organization}/${item.model} · ${formatCost(item.cost)} · ${item.accuracy.toFixed(2)}%`;
      tooltip.style.visibility = "hidden";
      const bounds = chartWrap.getBoundingClientRect();
      const pointX = event.clientX - bounds.left;
      const pointY = event.clientY - bounds.top;
      tooltip.style.left = `${Math.max(0, Math.min(pointX + 12, bounds.width - tooltip.offsetWidth))}px`;
      tooltip.style.top = `${Math.max(0, pointY - tooltip.offsetHeight - 12)}px`;
      tooltip.style.visibility = "visible";
    };
    const labelOffset = (index) => [-13, 21, -25, 29, -33, 37, -41][index % 7];
    const render = () => {
      const orgFiltered = data.filter((item) => selectedOrgs.has(item.organization));
      const frontierSource = orgFiltered.length ? orgFiltered : data;
      const optimal = frontier(frontierSource);
      const visible = paretoOnly.checked ? optimal : orgFiltered;
      const optimalIds = new Set(optimal.map((item) => `${item.organization}/${item.model}`));
      const scopeDescription = orgFiltered.length ? `${orgFiltered.length} model${orgFiltered.length === 1 ? "" : "s"} from selected organizations` : (paretoOnly.checked ? "No organizations selected — showing the all-model Pareto frontier" : "No organizations selected");
      status.textContent = `${scopeDescription} · ${optimal.length} frontier point${optimal.length === 1 ? "" : "s"}`;
      chart.replaceChildren();
      const title = svg("title", { id: "chart-title" }); title.textContent = "Cost versus ID accuracy";
      const description = svg("desc", { id: "chart-description" }); description.textContent = `${visible.length} visible models; color identifies organization. ${status.textContent}.`;
      chart.append(title, description);
      const bottom = dimensions.height - dimensions.bottom;
      const right = dimensions.width - dimensions.right;
      const axes = svg("g");
      [0.001, 0.01, 0.1, 1].filter((tick) => tick <= costDomain.max).forEach((tick) => {
        const tickX = x(tick); axes.append(svg("line", { class: "grid", x1: tickX, y1: dimensions.top, x2: tickX, y2: bottom }));
        const label = svg("text", { class: "tick", x: tickX, y: bottom + 22, "text-anchor": "middle" }); label.textContent = formatCost(tick); axes.append(label);
      });
      for (let tick = Math.ceil(accuracyDomain.min / 20) * 20; tick <= 100; tick += 20) {
        const tickY = y(tick); axes.append(svg("line", { class: "grid", x1: dimensions.left, y1: tickY, x2: right, y2: tickY }));
        const label = svg("text", { class: "tick", x: dimensions.left - 10, y: tickY + 4, "text-anchor": "end" }); label.textContent = `${tick}%`; axes.append(label);
      }
      axes.append(svg("line", { class: "axis", x1: dimensions.left, y1: bottom, x2: right, y2: bottom }), svg("line", { class: "axis", x1: dimensions.left, y1: dimensions.top, x2: dimensions.left, y2: bottom }));
      const xLabel = svg("text", { class: "axis-label", x: (dimensions.left + right) / 2, y: dimensions.height - 16, "text-anchor": "middle" }); xLabel.textContent = "Total cost (log scale)";
      const yLabel = svg("text", { class: "axis-label", x: 18, y: (dimensions.top + bottom) / 2, transform: `rotate(-90 18 ${(dimensions.top + bottom) / 2})`, "text-anchor": "middle" }); yLabel.textContent = "ID accuracy";
      axes.append(xLabel, yLabel); chart.append(axes);
      if (visible.length && optimal.length > 1) chart.append(svg("path", { class: "frontier-line", d: optimal.map((item, index) => `${index ? "L" : "M"}${x(item.cost)},${y(item.accuracy)}`).join(" ") }));
      const marks = svg("g");
      visible.forEach((item) => {
        const id = `${item.organization}/${item.model}`;
        const point = svg("circle", { class: `point${optimalIds.has(id) ? " frontier-point" : ""}${inspectedId === id ? " inspected" : ""}`, cx: x(item.cost), cy: y(item.accuracy), r: inspectedId === id ? 7 : optimalIds.has(id) ? 5.5 : 4, "aria-label": `${id}, ${formatCost(item.cost)}, ${item.accuracy.toFixed(2)} percent ID accuracy` });
        point.style.fill = orgColors.get(item.organization);
        point.addEventListener("mouseenter", (event) => { selectForInspection(item); showTooltip(event, item); });
        point.addEventListener("mousemove", (event) => showTooltip(event, item));
        point.addEventListener("mouseleave", hideTooltip);
        marks.append(point);
      });
      chart.append(marks);
      if (frontierLabels.checked && visible.length) {
        const labels = svg("g");
        optimal.forEach((item, index) => {
          const last = index === optimal.length - 1;
          const label = svg("text", { class: "frontier-label", x: x(item.cost) + (last ? -8 : 8), y: y(item.accuracy) + labelOffset(index), "text-anchor": last ? "end" : "start" });
          label.textContent = item.model; labels.append(label);
        });
        chart.append(labels);
      }
    };
    orgs.forEach((organization) => {
      const id = `org-${organization.replace(/[^a-z0-9]+/g, "-")}`;
      const label = document.createElement("label"); label.className = "organization"; label.style.setProperty("--org-color", orgColors.get(organization));
      const input = document.createElement("input"); input.id = id; input.type = "checkbox"; input.checked = true;
      input.addEventListener("change", () => { if (input.checked) selectedOrgs.add(organization); else selectedOrgs.delete(organization); hideTooltip(); render(); });
      const swatch = document.createElement("span"); swatch.className = "swatch"; swatch.setAttribute("aria-hidden", "true");
      const text = document.createElement("span"); text.textContent = organization;
      label.append(input, swatch, text); organizations.append(label);
      inspectOrg.append(new Option(organization, organization));
    });
    document.getElementById("select-all").addEventListener("click", () => { orgs.forEach((organization) => selectedOrgs.add(organization)); organizations.querySelectorAll("input").forEach((input) => { input.checked = true; }); render(); });
    document.getElementById("clear-all").addEventListener("click", () => { selectedOrgs.clear(); organizations.querySelectorAll("input").forEach((input) => { input.checked = false; }); hideTooltip(); render(); });
    paretoOnly.addEventListener("change", () => { hideTooltip(); render(); });
    frontierLabels.addEventListener("change", render);
    inspectOrg.addEventListener("change", () => { syncModelSelector(); setDetail(null); render(); });
    inspectModel.addEventListener("change", () => { const item = data.find((candidate) => candidate.organization === inspectOrg.value && candidate.model === inspectModel.value); setDetail(item || null); hideTooltip(); render(); });
    syncModelSelector();
    render();
  </script>
</body>
</html>
"""


def create_interactive_pareto_html(
    run_stats: Dict[str, Any],
    output_path: str | Path = "docs/pareto.html",
    title: str = "Model performance vs cost",
    y_metric: str = "id_top1",
) -> Path:
    """Write a self-contained interactive Pareto plot page."""

    points = _extract_points(run_stats, y_metric)
    if not points:
        raise ValueError("No models with positive cost and metric data were found")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(points, separators=(",", ":")).replace("</", "<\\/")
    html = _HTML_TEMPLATE.replace("__TITLE__", title).replace("__DATA__", data_json)
    output_file.write_text(html, encoding="utf-8")
    return output_file


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a standalone interactive Pareto plot")
    parser.add_argument("--runs-dir", default="tests/output/runs", help="Base directory for runs")
    parser.add_argument("--patterns", nargs="*", help="Patterns to filter runs")
    parser.add_argument("--doc-info", default="imgs/q11/doc_info.csv", help="Document info CSV")
    parser.add_argument("--test-ids", default="tests/data/test_ids.csv", help="Test IDs CSV")
    parser.add_argument("--output", default="docs/pareto.html", help="Output HTML file")
    parser.add_argument("--title", default="Model performance vs cost", help="Page title")
    parser.add_argument("--no-interactive", action="store_true", help="Skip interactive model review")
    parser.add_argument("--cohort-window-hours", type=float, default=24.0, help="Latest-cohort grouping window")
    parser.add_argument("--debug-cohorts", action="store_true", help="Print latest-cohort grouping details")
    args = parser.parse_args()

    generator = BenchmarkTableGenerator(args.runs_dir, interactive=not args.no_interactive)
    run_stats = generator.build_run_stats(
        run_patterns=args.patterns,
        doc_info_file=args.doc_info,
        test_ids_file=args.test_ids,
        cohort_window_hours=args.cohort_window_hours,
        debug_cohorts=args.debug_cohorts,
    )
    output_file = create_interactive_pareto_html(run_stats, args.output, args.title)
    print(f"Interactive Pareto plot saved to {output_file}")


if __name__ == "__main__":
    main()
