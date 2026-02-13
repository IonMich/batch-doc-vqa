# OpenRouter General Batch NER Plan

## Status Snapshot

- Completed: Phases 1-7
- Remaining: Phase 8 (synthetic PDF generator), Phase 9 (future UI), Phase 10 (future calibration experiment)

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
- [ ] Phase 8: Synthetic PDF task generator (PyMuPDF)
- [ ] Phase 9 (Future): Task-design UI workflow
- [ ] Phase 10 (Future): Statistical calibration plot experiment (OpenRouter, one-off script)

## Completed Scope (Phases 1-7)

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

## Remaining Scope

### Phase 8: Synthetic PDF task generator (PyMuPDF)

Core decision:

- Phase 8 v1 will be **template-driven and reproducible**, not a fully arbitrary layout engine.
- Goal is high practical coverage (forms, key-value regions, simple tables, noise), with deterministic generation and easy pipeline handoff.

Non-goals for v1:

- No WYSIWYG UI.
- No unconstrained free-form page design language.
- No photorealistic scan simulation; only controlled synthetic noise.

Primary deliverable:

- New one-off dataset generator CLI + script that produces benchmark-ready synthetic tasks.
- v1 targets compatibility with current benchmark tooling (`test_ids.csv` semantics for default student task).

Suggested implementation target:

- `src/batch_doc_vqa/tools/generate_synthetic_pdf_task.py`

### Phase 8.1: Data contract + CLI

Inputs:

- `--entities-file` (CSV/JSON with source values).
- `--task-config` (YAML/JSON defining target fields and layout profile).
- `--task-config` includes `task_type` (v1: `default_student`) and field mapping rules.
- `--output-dir`
- `--seed`
- Optional: `--num-docs`, `--pages-per-doc`, `--profile`.

Acceptance:

- Command validates schema and writes normalized `generation_plan.json`.
- Fails fast on invalid config or missing required fields.

### Phase 8.2: Layout primitives (deterministic)

Implement minimal reusable primitives:

- Header block
- Paragraph block
- Key-value block
- Simple table block (row/column text placement)

Acceptance:

- Deterministic box placement per seed.
- Overflow/collision detection with explicit errors.

### Phase 8.3: Target placement + distractor strategy

Generation behavior:

- Place target entities in configurable zones/pages.
- Add distractors with similar formats (for harder identification).
- Support multi-page docs where only subset of pages contain key fields.

Acceptance:

- Ground-truth rows map exactly to target entities per `doc`.
- Distractors never overwrite target truth values.

### Phase 8.4: Noise profiles (controlled realism)

Configurable noise:

- Extraneous text blocks
- Typographic variation (font family/size/weight)
- Position jitter
- Optional stamp-like overlays/watermark text

Acceptance:

- `clean`, `tabular`, `noisy_mixed` profiles generate visibly different difficulty.
- Same seed + same config reproduces identical generation plan.

### Phase 8.5: Artifact outputs

Required outputs:

- `task_docs.pdf`
- `test_ids.csv` (benchmark labels compatible with current scoring pipeline)
- `generation_plan.json` (doc/page/field placement trace)
- Optional `metadata.json` (seed, config hash, profile, generator version)

Acceptance:

- Artifacts are sufficient for downstream scoring and reproducibility audits.
- Current benchmark commands can consume outputs without scoring code changes.

### Phase 8.6: Pipeline handoff smoke path

Required end-to-end path:

1. `generate-synthetic-pdf-task`
2. `pdf-to-imgs`
3. `openrouter-inference`
4. benchmark table + Pareto plot

Acceptance:

- One documented smoke command sequence runs without source edits.
- Produced dataset can be benchmarked in current repo tooling.

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

Phase 8:

- Contract validation tests (`entities-file` + `task-config` failures).
- Determinism tests (same seed => same `generation_plan.json` and ground-truth rows).
- Layout tests (no collisions/overflow for valid fixtures; expected fail for invalid fixtures).
- Ground-truth alignment tests (doc/page/field mapping integrity).
- Profile tests (`clean`/`tabular`/`noisy_mixed` produce expected structural differences).
- Integration smoke test:
  `generate-synthetic-pdf-task -> pdf-to-imgs -> openrouter-inference -> benchmark`.

Phase 9:

- No implementation tests yet (design phase only).

Phase 10:

- Smoke test that script produces calibration plot from a labeled dataset.
- Verify plot artifact path and basic expected output shape.

## Suggested Next Order

1. Implement Phase 8.1 + 8.2 (contract + layout primitives).
2. Implement Phase 8.3 + 8.4 (targets/distractors + noise profiles).
3. Implement Phase 8.5 + 8.6 (artifacts + end-to-end smoke path).
4. Run one full synthetic end-to-end benchmark demo.
5. Plan Phase 9 as a separate UI design spike.
6. Prototype Phase 10 one-off OpenRouter calibration plot script.
