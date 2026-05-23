"""Versioned public synthetic benchmark suite generation."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

from ..tools.generate_synthetic_pdf_task import (
    PROFILE_DEFAULTS,
    TaskConfig,
    build_generation_plan,
    load_entities,
    write_generation_outputs,
)
from .pareto_plot import create_pareto_plot
from .table_generator import BenchmarkTableGenerator


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SUITE_SPEC = REPO_ROOT / "docs/examples/synthetic/default_student_suite_v1.yaml"
REQUIRED_ARTIFACTS = {
    "pdf",
    "labels",
    "generation_plan",
    "metadata",
    "normalized_entities",
    "doc_info",
    "benchmark_table",
    "pareto_plot",
    "id_lev_pareto_plot",
}


@dataclass(frozen=True)
class SuiteVariant:
    id: str
    profile: str
    seed: int
    num_docs: int
    pages_per_doc: int
    target_pages: tuple[int, ...]
    expected_artifacts: dict[str, str]


@dataclass(frozen=True)
class SuiteSpec:
    suite_id: str
    suite_version: int
    task_type: str
    entities_file: Path
    output_root: Path
    dpi: int
    variants: tuple[SuiteVariant, ...]
    spec_path: Path

    def variant_by_id(self, variant_id: str) -> SuiteVariant:
        for variant in self.variants:
            if variant.id == variant_id:
                return variant
        allowed = ", ".join(variant.id for variant in self.variants)
        raise ValueError(f"Unknown variant {variant_id!r}. Allowed: {allowed}")


def _resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _load_yaml_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Suite spec does not exist: {path}")
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"Suite spec must be a YAML object: {path}")
    return loaded


def _normalize_target_pages(raw: Any, *, pages_per_doc: int, variant_id: str) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Variant {variant_id}: target_pages must be a non-empty list")

    pages: list[int] = []
    seen: set[int] = set()
    for value in raw:
        try:
            page_num = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Variant {variant_id}: invalid target page {value!r}") from exc
        if page_num <= 0 or page_num > pages_per_doc:
            raise ValueError(
                f"Variant {variant_id}: target page {page_num} must be within 1..{pages_per_doc}"
            )
        if page_num not in seen:
            pages.append(page_num)
            seen.add(page_num)
    return tuple(pages)


def _parse_variant(raw: dict[str, Any]) -> SuiteVariant:
    variant_id = str(raw.get("id", "")).strip()
    if not variant_id:
        raise ValueError("Suite variant is missing id")

    profile = str(raw.get("profile", "")).strip()
    if profile not in PROFILE_DEFAULTS:
        allowed = ", ".join(sorted(PROFILE_DEFAULTS))
        raise ValueError(f"Variant {variant_id}: unsupported profile {profile!r}. Allowed: {allowed}")

    try:
        seed = int(raw["seed"])
        num_docs = int(raw["num_docs"])
        pages_per_doc = int(raw["pages_per_doc"])
    except KeyError as exc:
        raise ValueError(f"Variant {variant_id}: missing required field {exc.args[0]!r}") from exc
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Variant {variant_id}: seed, num_docs, and pages_per_doc must be integers") from exc

    if seed < 0:
        raise ValueError(f"Variant {variant_id}: seed must be >= 0")
    if num_docs <= 0:
        raise ValueError(f"Variant {variant_id}: num_docs must be >= 1")
    if pages_per_doc <= 0:
        raise ValueError(f"Variant {variant_id}: pages_per_doc must be >= 1")

    expected_artifacts = raw.get("expected_artifacts")
    if not isinstance(expected_artifacts, dict):
        raise ValueError(f"Variant {variant_id}: expected_artifacts must be an object")
    missing = REQUIRED_ARTIFACTS - set(expected_artifacts)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Variant {variant_id}: missing expected artifacts: {missing_text}")

    artifacts = {str(key): str(value) for key, value in expected_artifacts.items()}
    return SuiteVariant(
        id=variant_id,
        profile=profile,
        seed=seed,
        num_docs=num_docs,
        pages_per_doc=pages_per_doc,
        target_pages=_normalize_target_pages(
            raw.get("target_pages"),
            pages_per_doc=pages_per_doc,
            variant_id=variant_id,
        ),
        expected_artifacts=artifacts,
    )


def load_suite_spec(path: str | Path = DEFAULT_SUITE_SPEC) -> SuiteSpec:
    """Load and validate a versioned default_student synthetic suite spec."""
    spec_path = _resolve_repo_path(path)
    raw = _load_yaml_object(spec_path)

    suite_id = str(raw.get("suite_id", "")).strip()
    if not suite_id:
        raise ValueError("Suite spec is missing suite_id")

    task_type = str(raw.get("task_type", "")).strip()
    if task_type != "default_student":
        raise ValueError(f"Unsupported synthetic suite task_type {task_type!r}")

    try:
        suite_version = int(raw.get("suite_version", 0))
        dpi = int(raw.get("dpi", 150))
    except (TypeError, ValueError) as exc:
        raise ValueError("suite_version and dpi must be integers") from exc
    if suite_version <= 0:
        raise ValueError("suite_version must be >= 1")
    if dpi <= 0:
        raise ValueError("dpi must be >= 1")

    variants_raw = raw.get("variants")
    if not isinstance(variants_raw, list) or not variants_raw:
        raise ValueError("Suite spec must include at least one variant")

    variants: list[SuiteVariant] = []
    seen_ids: set[str] = set()
    seen_seeds: set[int] = set()
    for item in variants_raw:
        if not isinstance(item, dict):
            raise ValueError("Each suite variant must be an object")
        variant = _parse_variant(item)
        if variant.id in seen_ids:
            raise ValueError(f"Duplicate variant id: {variant.id}")
        if variant.seed in seen_seeds:
            raise ValueError(f"Duplicate variant seed: {variant.seed}")
        seen_ids.add(variant.id)
        seen_seeds.add(variant.seed)
        variants.append(variant)

    return SuiteSpec(
        suite_id=suite_id,
        suite_version=suite_version,
        task_type=task_type,
        entities_file=_resolve_repo_path(str(raw.get("entities_file", ""))),
        output_root=_resolve_repo_path(str(raw.get("output_root", ""))),
        dpi=dpi,
        variants=tuple(variants),
        spec_path=spec_path,
    )


def task_config_for_variant(variant: SuiteVariant) -> TaskConfig:
    """Build a default_student TaskConfig from a suite variant."""
    defaults = PROFILE_DEFAULTS[variant.profile]
    return TaskConfig(
        task_type="default_student",
        pages_per_doc=variant.pages_per_doc,
        target_pages=variant.target_pages,
        profile=variant.profile,
        include_table=bool(defaults["include_table"]),
        extra_lines_min=int(defaults["extra_lines_min"]),
        extra_lines_max=int(defaults["extra_lines_max"]),
        font_choices=tuple(str(font) for font in defaults["font_choices"]),
        jitter_px=float(defaults["jitter_px"]),
    )


def expected_artifact_paths(variant: SuiteVariant, variant_output_dir: Path) -> dict[str, Path]:
    return {
        artifact_name: variant_output_dir / relative_path
        for artifact_name, relative_path in variant.expected_artifacts.items()
    }


def render_pdf_to_images_deterministic(
    *,
    pdf_path: Path,
    pages_per_doc: int,
    output_dir: Path,
    dpi: int,
    filename_suffix: str,
) -> Path:
    """Render a suite PDF to images with stable filenames and doc_info.csv."""
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required to render suite images. Run with: "
            "uv run --with pymupdf generate-default-student-synthetic-suite"
        ) from exc

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pdf = fitz.open(str(pdf_path))
    try:
        if len(pdf) % pages_per_doc != 0:
            raise ValueError(
                f"PDF page count {len(pdf)} is not divisible by pages_per_doc={pages_per_doc}"
            )

        matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        doc_info_path = output_dir / "doc_info.csv"
        with open(doc_info_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["doc", "page", "filename"])
            for page_index, page in enumerate(pdf):
                doc_idx = page_index // pages_per_doc
                page_num = page_index % pages_per_doc + 1
                filename = f"doc-{doc_idx}-page-{page_num}-{filename_suffix}.png"
                page.get_pixmap(matrix=matrix).save(str(output_dir / filename))
                writer.writerow([doc_idx, page_num, filename])
    finally:
        pdf.close()

    return doc_info_path


def _clear_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if overwrite:
        shutil.rmtree(output_dir)
        return
    if any(output_dir.iterdir()):
        raise ValueError(f"Output directory already exists and is not empty: {output_dir}")


def _placeholder_benchmark_markdown(
    *,
    spec: SuiteSpec,
    variant: SuiteVariant,
    doc_info_path: Path,
    labels_path: Path,
) -> str:
    return "\n".join(
        [
            f"# {spec.suite_id} / {variant.id}",
            "",
            "No eligible model runs were found for this generated synthetic variant yet.",
            "",
            "This file is still useful as a reproducibility marker: the PDF, labels, image manifest, "
            "and expected report filenames were regenerated from the tracked suite spec.",
            "",
            "## Dataset",
            "",
            f"- Task: `{spec.task_type}`",
            f"- Profile: `{variant.profile}`",
            f"- Seed: `{variant.seed}`",
            f"- Documents: `{variant.num_docs}`",
            f"- Pages per doc: `{variant.pages_per_doc}`",
            f"- Target pages: `{', '.join(str(page) for page in variant.target_pages)}`",
            f"- `doc_info.csv`: `{doc_info_path}`",
            f"- Labels: `{labels_path}`",
            "",
            "Rerun this suite command after model inference writes matching run metadata.",
            "",
        ]
    )


def _write_placeholder_pareto(path: Path, *, title: str, message: str) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", wrap=True, fontsize=12)
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_variant_reports(
    *,
    spec: SuiteSpec,
    variant: SuiteVariant,
    variant_output_dir: Path,
    runs_dir: Path,
    run_patterns: list[str] | None,
) -> dict[str, Path]:
    artifacts = expected_artifact_paths(variant, variant_output_dir)
    doc_info_path = artifacts["doc_info"]
    labels_path = artifacts["labels"]
    benchmark_table_path = artifacts["benchmark_table"]
    pareto_plot_path = artifacts["pareto_plot"]
    id_lev_plot_path = artifacts["id_lev_pareto_plot"]

    generator = BenchmarkTableGenerator(str(runs_dir), interactive=False)
    run_stats = generator.build_run_stats(
        run_patterns=run_patterns,
        doc_info_file=str(doc_info_path),
        test_ids_file=str(labels_path),
    )

    benchmark_table_path.parent.mkdir(parents=True, exist_ok=True)
    if run_stats:
        table = generator._generate_markdown_table(run_stats, include_baseline=False)
        benchmark_table_path.write_text(
            "\n".join(
                [
                    f"# {spec.suite_id} / {variant.id}",
                    "",
                    f"- Task: `{spec.task_type}`",
                    f"- Profile: `{variant.profile}`",
                    f"- Seed: `{variant.seed}`",
                    f"- Documents: `{variant.num_docs}`",
                    "",
                    table,
                    "",
                ]
            ),
            encoding="utf-8",
        )
        create_pareto_plot(
            run_stats,
            str(pareto_plot_path),
            f"Model Performance vs Cost ({spec.suite_id}/{variant.id})",
        )
        create_pareto_plot(
            run_stats,
            str(id_lev_plot_path),
            f"Model ID Levenshtein Distance vs Cost ({spec.suite_id}/{variant.id})",
            y_metric="id_avg_lev",
            y_axis_label="ID Avg d_Lev (lower is better)",
            y_metric_print_label="avg ID d_Lev",
            y_metric_print_decimals=4,
            y_metric_suffix="",
            maximize_y=False,
            invert_y_axis=True,
        )
    else:
        benchmark_table_path.write_text(
            _placeholder_benchmark_markdown(
                spec=spec,
                variant=variant,
                doc_info_path=doc_info_path,
                labels_path=labels_path,
            ),
            encoding="utf-8",
        )
        message = "No eligible model runs found for this generated synthetic variant yet."
        _write_placeholder_pareto(
            pareto_plot_path,
            title=f"Model Performance vs Cost ({spec.suite_id}/{variant.id})",
            message=message,
        )
        _write_placeholder_pareto(
            id_lev_plot_path,
            title=f"Model ID Levenshtein Distance vs Cost ({spec.suite_id}/{variant.id})",
            message=message,
        )

    return {
        "benchmark_table": benchmark_table_path,
        "pareto_plot": pareto_plot_path,
        "id_lev_pareto_plot": id_lev_plot_path,
    }


def generate_variant(
    *,
    spec: SuiteSpec,
    variant: SuiteVariant,
    output_root: Path | None = None,
    overwrite: bool = False,
    runs_dir: Path | None = None,
    run_patterns: list[str] | None = None,
) -> dict[str, Path]:
    """Generate all configured artifacts for one suite variant."""
    root = output_root or spec.output_root
    variant_output_dir = root / variant.id
    _clear_output_dir(variant_output_dir, overwrite=overwrite)
    variant_output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_entities(spec.entities_file, seed=variant.seed, num_docs_limit=variant.num_docs)
    if len(rows) != variant.num_docs:
        raise ValueError(
            f"Variant {variant.id}: requested {variant.num_docs} docs but loaded {len(rows)} "
            f"from {spec.entities_file}"
        )

    config = task_config_for_variant(variant)
    plan = build_generation_plan(rows, config, seed=variant.seed)
    generated_paths = write_generation_outputs(
        output_dir=variant_output_dir,
        plan=plan,
        rows=rows,
        config=config,
        seed=variant.seed,
        entities_file=spec.entities_file,
        task_config_file=None,
        render_pdf=True,
    )

    artifacts = expected_artifact_paths(variant, variant_output_dir)
    doc_info_path = render_pdf_to_images_deterministic(
        pdf_path=artifacts["pdf"],
        pages_per_doc=variant.pages_per_doc,
        output_dir=artifacts["doc_info"].parent,
        dpi=spec.dpi,
        filename_suffix=variant.id,
    )

    report_paths = write_variant_reports(
        spec=spec,
        variant=variant,
        variant_output_dir=variant_output_dir,
        runs_dir=runs_dir or (REPO_ROOT / "tests/output/runs"),
        run_patterns=run_patterns,
    )

    generated_paths.update(
        {
            "labels": artifacts["labels"],
            "doc_info": doc_info_path,
            **report_paths,
        }
    )

    missing = [
        artifact_name
        for artifact_name, artifact_path in expected_artifact_paths(variant, variant_output_dir).items()
        if not artifact_path.exists()
    ]
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"Variant {variant.id}: missing expected generated artifacts: {missing_text}")

    return generated_paths


def generate_suite(
    *,
    spec: SuiteSpec,
    variant_ids: Iterable[str] | None = None,
    output_root: Path | None = None,
    overwrite: bool = False,
    runs_dir: Path | None = None,
    run_patterns: list[str] | None = None,
) -> dict[str, dict[str, Path]]:
    selected_variants = (
        [spec.variant_by_id(variant_id) for variant_id in variant_ids]
        if variant_ids is not None
        else list(spec.variants)
    )
    outputs: dict[str, dict[str, Path]] = {}
    for variant in selected_variants:
        outputs[variant.id] = generate_variant(
            spec=spec,
            variant=variant,
            output_root=output_root,
            overwrite=overwrite,
            runs_dir=runs_dir,
            run_patterns=run_patterns,
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate the versioned default_student synthetic benchmark suite.",
    )
    parser.add_argument(
        "--suite-spec",
        default=str(DEFAULT_SUITE_SPEC.relative_to(REPO_ROOT)),
        help="Suite YAML spec path.",
    )
    parser.add_argument(
        "--output-root",
        help="Override output root. Defaults to the output_root in the suite spec.",
    )
    parser.add_argument(
        "--variant",
        action="append",
        help="Variant id to regenerate. Repeat for multiple variants. Defaults to all variants.",
    )
    parser.add_argument(
        "--runs-dir",
        default="tests/output/runs",
        help="Run directory used when refreshing benchmark tables and Pareto plots.",
    )
    parser.add_argument(
        "--patterns",
        nargs="*",
        help="Optional run-name regex patterns passed to benchmark report generation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated suite output directories.",
    )
    args = parser.parse_args()

    spec = load_suite_spec(args.suite_spec)
    output_root = _resolve_repo_path(args.output_root) if args.output_root else None
    runs_dir = _resolve_repo_path(args.runs_dir)
    outputs = generate_suite(
        spec=spec,
        variant_ids=args.variant,
        output_root=output_root,
        overwrite=args.overwrite,
        runs_dir=runs_dir,
        run_patterns=args.patterns,
    )

    resolved_root = output_root or spec.output_root
    print(f"Generated {spec.suite_id} into {resolved_root}")
    for variant_id in sorted(outputs):
        print(f"- {variant_id}: {resolved_root / variant_id}")


if __name__ == "__main__":
    main()
