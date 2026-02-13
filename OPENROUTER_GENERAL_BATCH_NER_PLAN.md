# OpenRouter General Batch NER Plan

## Status Snapshot

- Completed: Phases 1-8
- Remaining: Phase 9 (future UI), Phase 10 (future calibration experiment)

## Goal

1. Bring your own scans/docs.
2. Bring your own extraction prompt.
3. Bring your own JSON schema.
4. Get structured JSON outputs without editing source code.
5. Generate synthetic test PDFs from chosen entities for rapid end-to-end benchmarking.

## Verification Checklist

- [x] Phase 1: CLI contract for custom extraction
- [x] Phase 2: Extraction spec loading
- [x] Phase 3: Schema-driven validation in inference
- [x] Phase 4: Parser generalization
- [x] Phase 5: Output ergonomics
- [x] Phase 6: Legacy benchmark compatibility
- [x] Phase 7: Documentation and examples
- [x] Phase 8: Synthetic PDF task generator (PyMuPDF)
- [ ] Phase 9 (Future): Task-design UI workflow
- [ ] Phase 10 (Future): Statistical calibration plot experiment (OpenRouter, one-off script)

## Completed Scope (Phases 1-8)

### Phase 1: CLI contract

- Added custom extraction flags in `openrouter-inference`:
  - `--prompt-file`
  - `--schema-file`
  - `--output-json`
  - `--strict-schema` / `--no-strict-schema`

### Phase 2: Extraction spec loading

- Added extraction spec loader:
  - `src/batch_doc_vqa/openrouter/spec.py`
- Supports preset defaults plus prompt/schema override files.

### Phase 3: Schema-driven validation

- Inference now validates against active schema.
- Supports strict failure and non-strict passthrough/coercion behavior.

### Phase 4: Parser generalization

- Parser keeps direct JSON + fenced JSON parsing.
- Generic object extraction fallback is in place.
- No student-specific regex fallback in parser path.

### Phase 5: Output ergonomics

- `--output-json` writes a stable external result copy.
- Run metadata includes extraction/preset/schema information.

### Phase 6: Legacy compatibility

- Default benchmark workflow remains available.
- Minimal run command remains supported.
- Default student preset + q11 defaults are preserved.

### Phase 7: Documentation and examples

- README restructured by workflow intent and code-change scope.
- Added custom prompt+schema example files under `docs/examples/...`.
- Added calibration material pointer in README and moved legacy details to `statistical-calibration.md`.

### Phase 8: Synthetic PDF task generator (PyMuPDF)

Delivered:

- New generator CLI: `uv run --with pymupdf generate-synthetic-pdf-task`.
- Implementation: `src/batch_doc_vqa/tools/generate_synthetic_pdf_task.py`.
- Example inputs:
  - `docs/examples/synthetic/default_student_entities.csv`
  - `docs/examples/synthetic/default_student_task_config.yaml`
- Unit coverage: `tests/test_synthetic_pdf_task_generator.py`.
- README workflow section for end-to-end synthetic benchmarking.

Verified end-to-end path:

1. `generate-synthetic-pdf-task`
2. `pdf-to-imgs`
3. `openrouter-inference`
4. `generate-benchmark-table` + `generate-pareto-plot`

## Remaining Scope (Future)

### Phase 9 (Future): Task-design UI workflow

Future direction:

- User describes task in plain language.
- System drafts prompt + schema (optionally with LLM assist).
- System generates PDFs, splits pages, runs inference, and produces evaluation artifacts.

### Phase 10 (Future): Statistical Calibration Plot Experiment (OpenRouter)

Future direction:

- Build a one-off calibration script (separate from `openrouter-inference` CLI surface).
- Reuse core request/parsing helpers where practical, but keep workflow isolated from main benchmark/extraction path.
- Run repeated stochastic inference for a fixed dataset and produce calibration plot artifacts comparable to existing `tests/output/public/calibration_curves.png`.
- Primary output goal: calibration plot generation, not a generalized calibration framework.

## Testing Plan (Remaining)

Phase 9:

- No implementation tests yet (design phase only).

Phase 10:

- Smoke test that script produces calibration plot from a labeled dataset.
- Verify plot artifact path and basic expected output shape.

## Suggested Next Order

1. Plan Phase 9 as a separate UI design spike.
2. Prototype Phase 10 one-off OpenRouter calibration plot script.
