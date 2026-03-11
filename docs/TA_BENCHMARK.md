# TA Benchmark v1

This document describes the TA benchmark tooling for instructor-pilot documents.

## Scope

- Tier 1: problem localization, description transcription, figure linkage, template matching.
- Tier 2: rubric-aligned scoring, error tags, feedback quality dimensions.
- Tier 3: deferred (LLM-as-judge without rubric, generated solution evaluation).

## Current Run Scope

- For the current execution phase (February 26, 2026), run Tier 1 only.
- Tier 2 is implemented but intentionally deferred until an explicit activation decision.

## Safety

- This workflow may process real student identifiers.
- Keep dataset artifacts local.
- Do not commit raw instructor dataset exports.

## Primary Commands

```bash
# Validate labels
uv run python -m batch_doc_vqa.ta_benchmark.validate_labels \
  --labels-dir /tmp/instructor_pilot_ta_v1/annotations

# Freeze dataset splits
uv run python -m batch_doc_vqa.ta_benchmark.dataset_freeze \
  --doc-info /tmp/instructor_pilot_export_full/doc_info.csv \
  --test-ids /tmp/instructor_pilot_export_full/test_ids.csv \
  --dataset-manifest /tmp/instructor_pilot_export_full/dataset_manifest.json \
  --output-root /tmp/instructor_pilot_ta_v1

# Generate model-assisted proposals
uv run python -m batch_doc_vqa.ta_benchmark.proposals \
  --results-json tests/output/runs/<run_name>/results.json \
  --doc-info /tmp/instructor_pilot_export_full/doc_info.csv \
  --test-ids /tmp/instructor_pilot_export_full/test_ids.csv \
  --output /tmp/instructor_pilot_ta_v1/proposals.json

# Launch annotation UI
uv run python -m batch_doc_vqa.datasets.ta_annotation_ui \
  --images-dir /tmp/instructor_pilot_export_full/images \
  --doc-info /tmp/instructor_pilot_export_full/doc_info.csv \
  --test-ids /tmp/instructor_pilot_export_full/test_ids.csv \
  --labels-dir /tmp/instructor_pilot_ta_v1/annotations \
  --proposals /tmp/instructor_pilot_ta_v1/proposals.json \
  --rubrics-dir /tmp/instructor_pilot_ta_v1/rubrics

# Score predictions against labels
uv run python -m batch_doc_vqa.ta_benchmark.score_runs \
  --labels-dir /tmp/instructor_pilot_ta_v1/annotations \
  --predictions tests/output/runs/<run_name>/results.json \
  --doc-info /tmp/instructor_pilot_export_full/doc_info.csv \
  --images-dir /tmp/instructor_pilot_export_full/images \
  --output /tmp/instructor_pilot_ta_v1/scores/<run_name>.json

# Generate markdown report
uv run python -m batch_doc_vqa.ta_benchmark.report \
  --scores-json /tmp/instructor_pilot_ta_v1/scores/<run_name>.json \
  --output-md /tmp/instructor_pilot_ta_v1/reports/<run_name>.md \
  --output-json /tmp/instructor_pilot_ta_v1/reports/<run_name>.summary.json

# Compare multiple runs and generate TA Pareto
uv run python -m batch_doc_vqa.ta_benchmark.compare_runs \
  --scores-glob "/tmp/instructor_pilot_ta_v1/scores/*.json" \
  --metric rubric_scoring.qwk \
  --cost-field run_metadata.cost_per_image \
  --output-md /tmp/instructor_pilot_ta_v1/reports/comparison.md \
  --pareto-output /tmp/instructor_pilot_ta_v1/reports/pareto_qwk.png

# Inter-annotator agreement check
uv run python -m batch_doc_vqa.ta_benchmark.agreement \
  --labels-a /tmp/instructor_pilot_ta_v1/annotations_reviewer_a \
  --labels-b /tmp/instructor_pilot_ta_v1/annotations_reviewer_b \
  --threshold-template 0.85 \
  --threshold-region-f1 0.80 \
  --output /tmp/instructor_pilot_ta_v1/reports/agreement.json
```

## Annotation Storage

- One file per document:
  - `/tmp/instructor_pilot_ta_v1/annotations/docs/doc-<id>.json`
- Schema:
  - `docs/examples/schemas/ta_benchmark_v1.schema.json`
- Error taxonomy:
  - `docs/examples/schemas/ta_error_taxonomy_v1.json`

## Notes

- `python -m batch_doc_vqa.ta_benchmark.score_runs` supports both normalized prediction JSON and image-keyed `results.json` when `--doc-info` is provided.
- Existing q11 benchmark commands are unchanged.
