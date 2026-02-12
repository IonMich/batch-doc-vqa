# OpenRouter General Batch NER Plan

## Status Snapshot

- Completed: Phases 1-6
- Remaining: Phase 7 (docs/examples), Phase 8 (synthetic PDF generator), Phase 9 (future UI), Phase 10 (future calibration experiment)

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
- [ ] Phase 7: Documentation and examples
- [ ] Phase 8: Synthetic PDF task generator (PyMuPDF)
- [ ] Phase 9 (Future): Task-design UI workflow
- [ ] Phase 10 (Future): Statistical calibration plot experiment (OpenRouter, one-off script)

## Completed Scope (Phases 1-6)

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

## Remaining Scope

### Phase 7: Documentation and examples

Deliverables:

- README section for custom extraction with prompt + schema + output JSON.
- Concrete example files (not placeholders), for example:
  - `docs/examples/prompts/basic-entity-extraction.md`
  - `docs/examples/schemas/basic-entity-extraction.schema.json`
- Copy-paste command in README using those files.
- Clear explanation of default benchmark mode vs custom mode.

Acceptance criteria:

- A first-time user can run custom extraction without editing source code.

### Phase 8: Synthetic PDF task generator (PyMuPDF)

Deliverable:

- New script + CLI entrypoint to generate benchmark-ready synthetic PDF tasks.

Suggested implementation target:

- `src/batch_doc_vqa/tools/generate_synthetic_pdf_task.py`

Required inputs:

- Entity rows (CSV/JSON).
- Field mapping to layout.
- Rendering options (font/noise/rotation/jitter/pages).
- Reproducibility seed.

Required outputs:

- Generated PDF batch (for example `/tmp/task/task_docs.pdf`).
- Ground-truth CSV aligned to generated docs.
- Optional generation metadata JSON.

Pipeline compatibility requirement:

1. `pdf-to-imgs`
2. `openrouter-inference`
3. benchmark table + Pareto plot generation

Acceptance criteria:

- Generated dataset can be evaluated end-to-end without code changes.

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

Phase 7:

- README command smoke-check with example prompt/schema files.

Phase 8:

- Determinism test (same seed => same outputs).
- Ground-truth alignment test (doc/page/file mapping integrity).
- Integration smoke test:
  `generate-synthetic-pdf-task -> pdf-to-imgs -> openrouter-inference`.

Phase 9:

- No implementation tests yet (design phase only).

Phase 10:

- Smoke test that script produces calibration plot from a labeled dataset.
- Verify plot artifact path and basic expected output shape.

## Suggested Next Order

1. Finish Phase 7 (docs/examples).
2. Implement Phase 8 (synthetic generator + tests).
3. Run one full synthetic end-to-end benchmark demo.
4. Plan Phase 9 as a separate UI design spike.
5. Prototype Phase 10 one-off OpenRouter calibration plot script.
