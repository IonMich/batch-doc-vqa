# TA Benchmark v1 Implementation Plan (Instructor Pilot)

## Summary

We will evolve the benchmark from ID/name/section extraction into a TA-relevant benchmark with auditable, evidence-grounded tasks.
Implementation is staged to produce usable outcomes early while controlling annotation cost and avoiding unbounded evaluation definitions.

## Current Execution Scope (Approved February 26, 2026)

- Active benchmark execution is Tier 1 only.
- Tier 2 annotation and Tier 2 scoring are deferred until an explicit go/no-go decision.
- `tier2_docs` and `dev_docs` are still frozen now as reserved subsets for later activation.

## Tier 1 Inference Strategy (Active)

- Do not treat Tier 1 as a single-prompt extraction task by default.
- Use a decomposed multi-step protocol for higher reliability and easier error attribution.
- Keep annotation-assist inference and benchmark-evaluation inference conceptually separate; they may converge later.

### Tier 1 Decomposed Protocol

1. Step A (page-level extraction):
- problem regions
- problem number candidates
- problem description text
- figure regions

2. Step B (doc-level consolidation):
- merge page-level outputs into canonical `problems[]`
- link figures to problem UIDs

3. Step C (doc-level template match):
- map consolidated problems to assignment template version

4. Step D (identity side-task, optional in same run):
- student name / ID / section extraction

### Protocol Selection Rule

- Evaluate both:
1. Protocol S: single-pass prompt (all Tier 1 fields).
2. Protocol M: decomposed multi-step protocol (recommended default).
- Run both on a fixed 30-doc pilot.
- Lock one protocol for the full frozen cohort before broad annotation.
- Benchmark comparisons must use one frozen protocol per track.

## Decisions Locked

1. Dataset source: instructor pilot submissions exported locally with `src/batch_doc_vqa/datasets/instructor_pilot.py`.
2. Benchmark tiers:
- Tier 1: problem localization, problem text transcription, figure linkage, assignment-template matching.
- Tier 2: rubric-aligned error tagging, rubric scoring, final per-problem grade, feedback quality dimensions.
- Tier 3: no-rubric LLM-as-judge and solution-generation scoring (deferred).
3. Evidence policy: every non-trivial label must include page/region evidence references.
4. Scope sizing:
- Tier 1 labels on full exported cohort (target: ~546 to ~600 docs, freeze by manifest hash).
- Tier 2 labels on 150-doc subset (reserved for later phase, not active now).
5. Annotation workflow: LLM proposal first, human verify/edit final.
6. Privacy: keep raw instructor data out of repo-tracked files; local-only dataset packs.

## In Scope

- Build annotation schema and tooling for Tier 1 and Tier 2.
- Add a writable annotation UI.
- Add scoring engine and reporting for new metrics.
- Add tests and docs for reproducible benchmark runs.

## Out of Scope (v1)

- Fully automated no-rubric grading.
- Ground-truthing generated reference solutions where official solutions are absent.
- Replacing human adjudication for contested grading decisions.

## Implementation Checklist

### Phase 0: Repo and Safety Setup
- [x] Add TA benchmark documentation skeleton under `docs/`.
- [x] Add `.gitignore` rules for local TA dataset and annotation artifacts.
- [x] Add explicit CLI warnings for PII-sensitive dataset operations.
- [x] Define local default workspace root for TA artifacts: `/tmp/instructor_pilot_ta_v1`.

### Phase 1: Benchmark Spec and Label Schema
- [x] Create schema module: `src/batch_doc_vqa/ta_benchmark/schema.py`.
- [x] Create JSON schema file: `docs/examples/schemas/ta_benchmark_v1.schema.json`.
- [x] Define error taxonomy file: `docs/examples/schemas/ta_error_taxonomy_v1.json`.
- [x] Define rubric schema file: `docs/examples/schemas/ta_rubric_v1.schema.json`.
- [x] Add validation CLI: `uv run python -m batch_doc_vqa.ta_benchmark.validate_labels --labels-dir ... --schema ...`.

### Phase 2: Dataset Freeze and Cohort Manifest
- [x] Add dataset freeze CLI (`ta-freeze-dataset`) for reproducible cohort metadata.
- [x] Materialize frozen dataset manifest and record `dataset_hash` (CLI output).
- [x] Emit `cohort_definition.json` with assignment IDs and filter policy used (CLI output).
- [x] Add dataset metadata file `ta_dataset.yaml` with split definitions:
- [x] `tier1_docs`: full list.
- [x] `tier2_docs`: fixed 150-doc subset.
- [x] `dev_docs`: 30-doc calibration set.
- [x] `test_docs`: remaining docs by split policy.
- [ ] Run instructor-pilot export dry-run and apply export to local TA pack (operator step).

### Phase 3: Annotation UI MVP (Tier 1)
- [x] Implement new UI script: `src/batch_doc_vqa/datasets/ta_annotation_ui.py`.
- [x] Reuse layout patterns from `inspect_ground_truth_ui.py`.
- [x] Add editable overlays for problem and figure bounding boxes.
- [x] Add editable fields for problem number, description text, template version.
- [x] Add evidence reference management (auto IDs per region).
- [x] Add Save, Auto-save, and Resume by doc ID.
- [x] Persist one file per doc in `annotations/docs/doc-<id>.json`.
- [x] Add keyboard shortcuts: next doc, prev doc, save, add box.

### Phase 4: LLM Proposal Pipeline
- [x] Add proposal generator CLI: `uv run python -m batch_doc_vqa.ta_benchmark.proposals --results-json ... --output ...`.
- [x] Proposal payload includes Tier 1 fields and confidence hints.
- [x] UI loads proposal and tracks reviewer action:
- [x] `accepted`.
- [x] `edited`.
- [x] `rejected`.
- [x] Store proposal provenance in each label file.
- [ ] Add Tier 1 protocol tracks:
- [ ] `single_pass_tier1` (Protocol S).
- [ ] `decomposed_tier1` (Protocol M).
- [ ] Run 30-doc pilot comparison and freeze selected protocol.

### Phase 5: Tier 2 Annotation Features (Implemented, Execution Deferred)
- [x] Add rubric loader in UI from `rubrics/<assignment_id>.yaml`.
- [x] Add rubric scoring form per problem criterion.
- [x] Add error-tag picker (string/tag fields constrained by taxonomy at save-time validation).
- [x] Add per-problem feedback field with actionability flag.
- [x] Add validation constraints:
- [x] awarded points must be in `[0, max_points]`.
- [x] each deduction must cite at least one evidence ref.
- [x] feedback must link to at least one problem.

### Phase 6: Scoring Engine and Metrics (Tier 1 Active, Tier 2 Deferred)
- [x] Implement scorer module: `src/batch_doc_vqa/ta_benchmark/metrics.py`.
- [x] Implement run evaluator CLI: `uv run python -m batch_doc_vqa.ta_benchmark.score_runs --labels-dir ... --predictions ... --output ...`.
- [x] Implement report generator CLI: `uv run python -m batch_doc_vqa.ta_benchmark.report --scores-json ... --output-md ... --output-json ...`.
- [x] Tier 1 metrics:
- [x] Problem region detection: precision, recall, F1, IoU@0.5.
- [x] Description transcription: CER and normalized edit distance.
- [x] Figure association: link precision, recall, F1.
- [x] Template matching: top-1 accuracy.
- [x] Tier 2 metrics:
- [x] Error tagging: micro/macro F1.
- [x] Rubric scoring: MAE, exact match %, QWK.
- [x] Feedback: correctness/specificity/actionability agreement.

### Phase 7: Benchmark Integration and Reporting
- [x] Add TA benchmark docs and command flow documentation.
- [x] Keep existing q11 benchmark untouched and runnable.
- [x] Add dataset-scoped TA report artifact path guidance (`tests/output/ta_runs/` or local `/tmp`).
- [x] Generate model comparison table + TA Pareto support via `ta-compare-runs`.

### Phase 8: Tests and Acceptance Gates
- [x] Add schema validation tests: `tests/test_ta_schema.py`.
- [x] Add UI data I/O tests: `tests/test_ta_annotation_io.py`.
- [x] Add metric correctness tests with hand-checked fixtures: `tests/test_ta_metrics.py`.
- [x] Add end-to-end smoke test on 20-doc fixture: `tests/test_ta_smoke.py`.
- [x] Add agreement calibration script and threshold checks (`ta-check-agreement`).
- [ ] Require acceptance gates before full run publication (operational policy step).

### Phase 9: Deferred Backlog (Post-v1)
- [ ] No-rubric LLM-as-judge evaluation track.
- [ ] Generated-solution evaluation track.
- [ ] Multi-annotator disagreement adjudication dashboard.

## Immediate Execution Order (Tier 1 Only)

1. Complete Phase 0 through Phase 4 operational steps on the instructor pilot export.
2. Run Protocol S vs Protocol M on a fixed 30-doc Tier 1 pilot and freeze selected protocol.
3. Finish Tier 1 labeling for the frozen full cohort using the frozen protocol.
4. Run validation and Tier 1 scoring/report generation.
5. Review Tier 1 quality and annotation throughput.
6. Decide whether to activate Tier 2.
