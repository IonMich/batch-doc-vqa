#!/usr/bin/env python3
"""
Generate deterministic synthetic PDF tasks for benchmark smoke runs.

Phase 8 v1 intentionally uses template-driven layouts:
- deterministic placement
- configurable profile difficulty
- benchmark-compatible labels (test_ids.csv)
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import yaml


PROFILE_DEFAULTS: dict[str, dict[str, Any]] = {
    "clean": {
        "include_table": False,
        "extra_lines_min": 0,
        "extra_lines_max": 1,
        "font_choices": ("helv",),
        "jitter_px": 0.0,
    },
    "tabular": {
        "include_table": True,
        "extra_lines_min": 1,
        "extra_lines_max": 3,
        "font_choices": ("helv", "cour"),
        "jitter_px": 0.5,
    },
    "noisy_mixed": {
        "include_table": True,
        "extra_lines_min": 3,
        "extra_lines_max": 7,
        "font_choices": ("helv", "cour", "times-roman"),
        "jitter_px": 2.0,
    },
}

_PAGE_SIZE_A4 = (595.0, 842.0)
_NOISE_TEMPLATES = (
    "Submission batch code: {code}",
    "Reviewed by assistant marker {code}",
    "Archive reference: REF-{code}",
    "Classroom note: desk rotation {num}",
    "Administrative note: copy {num}",
    "Department memo {code}: see annex",
    "Exam packet index {num} / section {sec}",
)


@dataclass(frozen=True)
class TaskConfig:
    task_type: str
    pages_per_doc: int
    target_pages: tuple[int, ...]
    profile: str
    include_table: bool
    extra_lines_min: int
    extra_lines_max: int
    font_choices: tuple[str, ...]
    jitter_px: float


@dataclass(frozen=True)
class EntityRow:
    doc: int
    student_full_name: str
    student_id: str
    section_number: str


def _load_structured_file(path: Path) -> Any:
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")
    raw = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(raw)
    if suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(raw)
        return {} if data is None else data
    raise ValueError(f"Unsupported config format for {path}. Use .json, .yaml, or .yml")


def _normalize_digits(value: Optional[str], *, target_len: int, field_name: str, required: bool) -> str:
    if value is None:
        value = ""
    normalized = str(value).strip()
    if not normalized:
        if required:
            raise ValueError(f"Missing required numeric field: {field_name}")
        return ""
    if not normalized.isdigit():
        raise ValueError(f"Invalid {field_name}={value!r}: expected digits only")
    if len(normalized) > target_len:
        raise ValueError(f"Invalid {field_name}={value!r}: exceeds max length {target_len}")
    return normalized.zfill(target_len)


def load_task_config(
    config_path: Optional[Path],
    *,
    pages_per_doc_override: Optional[int] = None,
    profile_override: Optional[str] = None,
) -> TaskConfig:
    raw: dict[str, Any] = {}
    if config_path is not None:
        loaded = _load_structured_file(config_path)
        if not isinstance(loaded, dict):
            raise ValueError("Task config must be an object")
        raw = dict(loaded)

    profile = str(profile_override or raw.get("profile", "clean")).strip()
    if profile not in PROFILE_DEFAULTS:
        allowed = ", ".join(sorted(PROFILE_DEFAULTS))
        raise ValueError(f"Unsupported profile {profile!r}. Allowed: {allowed}")
    profile_defaults = PROFILE_DEFAULTS[profile]

    task_type = str(raw.get("task_type", "default_student")).strip()
    if task_type != "default_student":
        raise ValueError(
            f"Unsupported task_type {task_type!r}. Phase 8 v1 supports only 'default_student'."
        )

    pages_per_doc_raw = pages_per_doc_override or raw.get("pages_per_doc", 4)
    try:
        pages_per_doc = int(pages_per_doc_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid pages_per_doc={pages_per_doc_raw!r}") from exc
    if pages_per_doc <= 0:
        raise ValueError("pages_per_doc must be >= 1")

    target_pages_raw = raw.get("target_pages", [1, 3])
    if not isinstance(target_pages_raw, (list, tuple)) or not target_pages_raw:
        raise ValueError("target_pages must be a non-empty list of positive integers")
    normalized_target_pages: list[int] = []
    seen: set[int] = set()
    for value in target_pages_raw:
        try:
            page_num = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid target page value {value!r}") from exc
        if page_num <= 0:
            raise ValueError("target_pages values must be >= 1")
        if page_num > pages_per_doc:
            raise ValueError(
                f"target_pages contains {page_num}, which exceeds pages_per_doc={pages_per_doc}"
            )
        if page_num in seen:
            continue
        seen.add(page_num)
        normalized_target_pages.append(page_num)

    include_table = bool(raw.get("include_table", profile_defaults["include_table"]))
    extra_lines_min = int(raw.get("extra_lines_min", profile_defaults["extra_lines_min"]))
    extra_lines_max = int(raw.get("extra_lines_max", profile_defaults["extra_lines_max"]))
    if extra_lines_min < 0 or extra_lines_max < 0 or extra_lines_min > extra_lines_max:
        raise ValueError("extra_lines_min/extra_lines_max must satisfy 0 <= min <= max")

    font_choices_raw = raw.get("font_choices", profile_defaults["font_choices"])
    if not isinstance(font_choices_raw, (list, tuple)) or not font_choices_raw:
        raise ValueError("font_choices must be a non-empty list")
    font_choices = tuple(str(font).strip() for font in font_choices_raw if str(font).strip())
    if not font_choices:
        raise ValueError("font_choices must contain at least one non-empty value")

    jitter_px_raw = raw.get("jitter_px", profile_defaults["jitter_px"])
    jitter_px = float(jitter_px_raw)
    if jitter_px < 0:
        raise ValueError("jitter_px must be >= 0")

    return TaskConfig(
        task_type=task_type,
        pages_per_doc=pages_per_doc,
        target_pages=tuple(normalized_target_pages),
        profile=profile,
        include_table=include_table,
        extra_lines_min=extra_lines_min,
        extra_lines_max=extra_lines_max,
        font_choices=font_choices,
        jitter_px=jitter_px,
    )


def _iter_entity_dicts(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise ValueError(f"Entities file does not exist: {path}")

    suffix = path.suffix.lower()
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield dict(row)
        return

    if suffix == ".json":
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(loaded, list):
            for row in loaded:
                if not isinstance(row, dict):
                    raise ValueError("Entities JSON list must contain objects")
                yield dict(row)
            return
        raise ValueError("Entities JSON must be a list of objects")

    raise ValueError(f"Unsupported entities format for {path}. Use .csv or .json")


def _random_digits(rng: random.Random, length: int) -> str:
    # Avoid all-zero IDs in synthetic data.
    first = str(rng.randint(1, 9))
    rest = "".join(str(rng.randint(0, 9)) for _ in range(length - 1))
    return first + rest


def load_entities(
    entities_path: Path,
    *,
    seed: int,
    num_docs_limit: Optional[int] = None,
) -> list[EntityRow]:
    rng = random.Random(seed + 17)
    rows: list[EntityRow] = []

    for index, raw in enumerate(_iter_entity_dicts(entities_path)):
        name = str(raw.get("student_full_name") or raw.get("name") or "").strip()
        if not name:
            raise ValueError(
                f"Missing student_full_name/name at entity row index {index} in {entities_path}"
            )

        doc_raw = raw.get("doc")
        if doc_raw is None or str(doc_raw).strip() == "":
            doc = index
        else:
            try:
                doc = int(doc_raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid doc index {doc_raw!r} at row {index}") from exc
            if doc < 0:
                raise ValueError(f"doc index must be >= 0; got {doc}")

        student_id_raw = (
            raw.get("student_id")
            or raw.get("university_id")
            or raw.get("ufid")
            or ""
        )
        student_id = _normalize_digits(
            str(student_id_raw),
            target_len=8,
            field_name="student_id",
            required=False,
        )
        if not student_id:
            student_id = _random_digits(rng, 8)

        section_raw = raw.get("section_number") or ""
        section_number = _normalize_digits(
            str(section_raw),
            target_len=5,
            field_name="section_number",
            required=False,
        )
        if not section_number:
            section_number = _random_digits(rng, 5)

        rows.append(
            EntityRow(
                doc=doc,
                student_full_name=name,
                student_id=student_id,
                section_number=section_number,
            )
        )

    if not rows:
        raise ValueError(f"No entities loaded from {entities_path}")

    # Keep stable ordering by doc then original order.
    rows.sort(key=lambda item: item.doc)

    if num_docs_limit is not None:
        if num_docs_limit <= 0:
            raise ValueError("num_docs_limit must be >= 1")
        rows = rows[:num_docs_limit]

    # Re-index docs densely from 0 for benchmark compatibility.
    remapped: list[EntityRow] = []
    for idx, row in enumerate(rows):
        remapped.append(
            EntityRow(
                doc=idx,
                student_full_name=row.student_full_name,
                student_id=row.student_id,
                section_number=row.section_number,
            )
        )
    return remapped


def _noise_text(rng: random.Random, section_number: str) -> str:
    template = rng.choice(_NOISE_TEMPLATES)
    return template.format(
        code="".join(rng.choice("ABCDEFGHJKLMNPQRSTUVWXYZ23456789") for _ in range(6)),
        num=rng.randint(10, 999),
        sec=section_number,
    )


def _apply_jitter(value: float, *, jitter_px: float, rng: random.Random) -> float:
    if jitter_px <= 0:
        return value
    return value + rng.uniform(-jitter_px, jitter_px)


def build_generation_plan(
    entities: list[EntityRow],
    config: TaskConfig,
    *,
    seed: int,
) -> dict[str, Any]:
    rng = random.Random(seed + 29)
    docs: list[dict[str, Any]] = []

    for entity in entities:
        pages: list[dict[str, Any]] = []
        for page_num in range(1, config.pages_per_doc + 1):
            elements: list[dict[str, Any]] = []
            font = rng.choice(config.font_choices)

            # Header line on every page for document realism.
            elements.append(
                {
                    "type": "text",
                    "x": _apply_jitter(72.0, jitter_px=config.jitter_px, rng=rng),
                    "y": _apply_jitter(52.0, jitter_px=config.jitter_px, rng=rng),
                    "font": font,
                    "size": 10.5,
                    "text": f"Physics Quiz Submission Packet - Document {entity.doc}",
                }
            )

            # Primary target fields for default student extraction.
            if page_num == 1:
                elements.extend(
                    [
                        {
                            "type": "text",
                            "x": _apply_jitter(72.0, jitter_px=config.jitter_px, rng=rng),
                            "y": _apply_jitter(118.0, jitter_px=config.jitter_px, rng=rng),
                            "font": font,
                            "size": 12.0,
                            "text": f"Student Name: {entity.student_full_name}",
                        },
                        {
                            "type": "text",
                            "x": _apply_jitter(72.0, jitter_px=config.jitter_px, rng=rng),
                            "y": _apply_jitter(148.0, jitter_px=config.jitter_px, rng=rng),
                            "font": font,
                            "size": 12.0,
                            "text": f"Section Number: {entity.section_number}",
                        },
                    ]
                )

            if page_num in config.target_pages:
                id_y = 218.0 if page_num == 1 else 128.0
                elements.append(
                    {
                        "type": "text",
                        "x": _apply_jitter(72.0, jitter_px=config.jitter_px, rng=rng),
                        "y": _apply_jitter(id_y, jitter_px=config.jitter_px, rng=rng),
                        "font": font,
                        "size": 12.0,
                        "text": f"University ID: {entity.student_id}",
                    }
                )

                if config.include_table:
                    rows = [
                        ["Field", "Value", "Notes"],
                        ["Student", entity.student_full_name, "candidate"],
                        ["UFID", entity.student_id, "verified"],
                        ["Section", entity.section_number, "recorded"],
                    ]
                    elements.append(
                        {
                            "type": "table",
                            "x": _apply_jitter(72.0, jitter_px=config.jitter_px, rng=rng),
                            "y": _apply_jitter(300.0, jitter_px=config.jitter_px, rng=rng),
                            "col_widths": [110.0, 220.0, 120.0],
                            "row_height": 22.0,
                            "font": font,
                            "size": 9.5,
                            "rows": rows,
                        }
                    )

            noise_count = rng.randint(config.extra_lines_min, config.extra_lines_max)
            y_start = 420.0
            for idx in range(noise_count):
                elements.append(
                    {
                        "type": "text",
                        "x": _apply_jitter(72.0, jitter_px=config.jitter_px, rng=rng),
                        "y": _apply_jitter(y_start + idx * 22.0, jitter_px=config.jitter_px, rng=rng),
                        "font": rng.choice(config.font_choices),
                        "size": 9.5,
                        "text": _noise_text(rng, entity.section_number),
                    }
                )

            pages.append({"page": page_num, "elements": elements})

        docs.append(
            {
                "doc": entity.doc,
                "target": {
                    "student_full_name": entity.student_full_name,
                    "student_id": entity.student_id,
                    "section_number": entity.section_number,
                },
                "pages": pages,
            }
        )

    return {
        "version": "1",
        "task_type": config.task_type,
        "profile": config.profile,
        "seed": seed,
        "pages_per_doc": config.pages_per_doc,
        "target_pages": list(config.target_pages),
        "docs": docs,
        "page_size": "A4",
    }


def _draw_table(page: Any, element: dict[str, Any]) -> None:
    x = float(element["x"])
    y = float(element["y"])
    col_widths = [float(v) for v in element["col_widths"]]
    row_height = float(element["row_height"])
    rows = element["rows"]
    font = str(element.get("font", "helv"))
    size = float(element.get("size", 10.0))
    n_rows = len(rows)

    table_w = sum(col_widths)
    table_h = row_height * n_rows
    page.draw_rect((x, y, x + table_w, y + table_h), width=0.8, color=(0, 0, 0))

    current_x = x
    for width in col_widths[:-1]:
        current_x += width
        page.draw_line((current_x, y), (current_x, y + table_h), width=0.6, color=(0, 0, 0))

    current_y = y
    for _ in range(n_rows - 1):
        current_y += row_height
        page.draw_line((x, current_y), (x + table_w, current_y), width=0.6, color=(0, 0, 0))

    for row_idx, row in enumerate(rows):
        row_y = y + row_idx * row_height + 14.0
        col_x = x + 6.0
        for col_idx, value in enumerate(row):
            _ = col_idx
            page.insert_text(
                (col_x, row_y),
                str(value),
                fontsize=size,
                fontname=font,
                color=(0, 0, 0),
            )
            col_x += col_widths[col_idx]


def render_plan_to_pdf(plan: dict[str, Any], output_pdf_path: Path) -> None:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required to render PDFs. Run with: uv run --with pymupdf generate-synthetic-pdf-task ..."
        ) from exc

    width, height = _PAGE_SIZE_A4
    pdf = fitz.open()
    for doc_entry in plan["docs"]:
        for page_entry in doc_entry["pages"]:
            page = pdf.new_page(width=width, height=height)
            for element in page_entry["elements"]:
                element_type = element.get("type")
                if element_type == "text":
                    page.insert_text(
                        (float(element["x"]), float(element["y"])),
                        str(element["text"]),
                        fontsize=float(element.get("size", 10.0)),
                        fontname=str(element.get("font", "helv")),
                        color=(0, 0, 0),
                    )
                elif element_type == "table":
                    _draw_table(page, element)

    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.save(str(output_pdf_path))
    pdf.close()


def write_test_ids_csv(rows: list[EntityRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["doc", "student_id", "student_full_name"])
        for row in rows:
            writer.writerow([row.doc, row.student_id, row.student_full_name])


def write_generation_outputs(
    *,
    output_dir: Path,
    plan: dict[str, Any],
    rows: list[EntityRow],
    config: TaskConfig,
    seed: int,
    entities_file: Path,
    task_config_file: Optional[Path],
    render_pdf: bool,
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    plan_path = output_dir / "generation_plan.json"
    plan_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["generation_plan"] = plan_path

    labels_path = output_dir / "test_ids.csv"
    write_test_ids_csv(rows, labels_path)
    paths["test_ids"] = labels_path

    metadata = {
        "version": "1",
        "seed": seed,
        "task_type": config.task_type,
        "profile": config.profile,
        "pages_per_doc": config.pages_per_doc,
        "target_pages": list(config.target_pages),
        "entities_file": str(entities_file.resolve(strict=False)),
        "task_config_file": str(task_config_file.resolve(strict=False)) if task_config_file else None,
        "num_docs": len(rows),
        "render_pdf": render_pdf,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["metadata"] = metadata_path

    if render_pdf:
        pdf_path = output_dir / "task_docs.pdf"
        render_plan_to_pdf(plan, pdf_path)
        paths["pdf"] = pdf_path

    # Helpful normalized copy for quick inspection.
    normalized_rows = [asdict(row) for row in rows]
    normalized_path = output_dir / "entities.normalized.json"
    normalized_path.write_text(json.dumps(normalized_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["normalized_entities"] = normalized_path
    return paths


def _ensure_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if not output_dir.exists():
        return
    if not overwrite:
        existing_files = [p for p in output_dir.iterdir() if p.is_file()]
        if existing_files:
            raise ValueError(
                f"Output directory {output_dir} already contains files. "
                "Use --overwrite to reuse it."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic PDF tasks (Phase 8 v1, template-driven).",
    )
    parser.add_argument(
        "--entities-file",
        type=str,
        required=True,
        help="CSV/JSON source rows. Columns for v1: student_full_name,name,student_id,university_id,ufid,section_number (doc optional).",
    )
    parser.add_argument(
        "--task-config",
        type=str,
        help="YAML/JSON task config. v1 task_type must be 'default_student'.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for task_docs.pdf, test_ids.csv, generation_plan.json, metadata.json.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Deterministic generation seed (default: 42).")
    parser.add_argument("--num-docs", type=int, help="Optional cap on number of generated docs.")
    parser.add_argument("--pages-per-doc", type=int, help="Override pages_per_doc from task config.")
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_DEFAULTS.keys()),
        help="Difficulty profile override.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--skip-pdf",
        action="store_true",
        help="Write plan/labels/metadata only (no PDF render).",
    )

    args = parser.parse_args()

    entities_file = Path(args.entities_file).expanduser().resolve(strict=False)
    task_config_file = (
        Path(args.task_config).expanduser().resolve(strict=False)
        if args.task_config
        else None
    )
    output_dir = Path(args.output_dir).expanduser().resolve(strict=False)

    _ensure_output_dir(output_dir, overwrite=args.overwrite)

    config = load_task_config(
        task_config_file,
        pages_per_doc_override=args.pages_per_doc,
        profile_override=args.profile,
    )
    rows = load_entities(
        entities_file,
        seed=args.seed,
        num_docs_limit=args.num_docs,
    )
    plan = build_generation_plan(rows, config, seed=args.seed)
    paths = write_generation_outputs(
        output_dir=output_dir,
        plan=plan,
        rows=rows,
        config=config,
        seed=args.seed,
        entities_file=entities_file,
        task_config_file=task_config_file,
        render_pdf=not args.skip_pdf,
    )

    print(f"Generated {len(rows)} synthetic docs using profile '{config.profile}'.")
    print(f"- generation plan: {paths['generation_plan']}")
    print(f"- labels (test_ids): {paths['test_ids']}")
    print(f"- metadata: {paths['metadata']}")
    print(f"- normalized entities: {paths['normalized_entities']}")
    if "pdf" in paths:
        print(f"- task PDF: {paths['pdf']}")
        print("\nNext steps:")
        print(
            "uv run --with pymupdf pdf-to-imgs "
            f"--filepath {paths['pdf']} --pages_i {config.pages_per_doc} --dpi 300 --output_dir {output_dir / 'images'}"
        )
    else:
        print("- task PDF: skipped (--skip-pdf)")


if __name__ == "__main__":
    main()

