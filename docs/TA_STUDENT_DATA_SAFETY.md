# TA Student Data Safety Runbook

Last audited: 2026-05-10

This runbook defines the safety boundary for continuing the Instructor Pilot TA benchmark work. It is intentionally narrower than the general benchmark docs: it exists to prevent raw student submissions, identifiers, grades, comments, or derived labels from being exposed while still allowing the benchmark tooling to move forward.

## Objective

Make the TA benchmark safe to continue by keeping public-safe repo work separate from private local student-data operations.

Success means:

- code, schemas, prompts, docs, and synthetic fixtures remain public-safe;
- real Instructor Pilot exports stay local and ignored by git;
- Codex or other assistants can inspect code and aggregate command output, but must not read, quote, copy, summarize, or upload raw student artifacts;
- Tier 1 protocol S/M pilot work has a concrete checklist that preserves those boundaries.

## Current Safety Evidence

The existing repo already has concrete safety scaffolding:

- `.gitignore` excludes `ta_artifacts/`, `annotations/`, and `tests/output/ta_runs/`.
- `.gitignore` also excludes `tests/output/*`, which covers local model runs under `tests/output/runs/`.
- `src/batch_doc_vqa/ta_benchmark/constants.py` defines a PII warning banner for TA workflows.
- `docs/TA_BENCHMARK.md` states that real student identifiers may be processed, dataset artifacts should stay local, and raw instructor exports should not be committed.
- `docs/TA_BENCHMARK_PLAN.md` keeps active execution to Tier 1, defers Tier 2 until explicit approval, and requires local-only dataset packs.
- `src/batch_doc_vqa/datasets/instructor_pilot.py` writes exports containing student identifiers and submission metadata, so those outputs must be treated as private.
- `src/batch_doc_vqa/ta_benchmark/proposals.py` can write model proposals with identity guesses, so proposal files are private unless explicitly anonymized and reviewed.
- `src/batch_doc_vqa/datasets/ta_annotation_ui.py` persists per-document labels with review state, evidence regions, problems, and optional rubric/feedback fields, so annotation outputs are private by default.

Boundary checks used for this audit:

```bash
git -C /Users/ioannism/repos/batch-doc-vqa check-ignore -v annotations/ tests/output/ta_runs/ ta_artifacts/
git -C /Users/ioannism/repos/batch-doc-vqa status --ignored --short ta_artifacts annotations tests/output/ta_runs tests/output/runs tests/output/public
git -C /Users/ioannism/repos/batch-doc-vqa ls-files
```

The tracked-file audit found TA code, schemas, prompts, and docs tracked, but no tracked `ta_artifacts/`, `annotations/`, `tests/output/ta_runs/`, `submission_metadata.csv`, `doc_info.csv`, or private Instructor Pilot workspace files. The tracked `tests/data/test_ids.csv` belongs to the built-in q11 benchmark fixture, not the local Instructor Pilot export.

## Public-Safe Material

These may be inspected, edited, committed, or referenced in public docs:

- `README.md`, `BENCHMARKS.md`, and general benchmark docs.
- `docs/TA_BENCHMARK.md`, `docs/TA_BENCHMARK_PLAN.md`, `docs/TA_TIER1_PROTOCOLS.md`, and this runbook.
- Prompt templates under `docs/examples/prompts/`.
- JSON schemas and taxonomies under `docs/examples/schemas/`.
- Synthetic example config and entity fixtures under `docs/examples/synthetic/`.
- TA benchmark source code under `src/batch_doc_vqa/ta_benchmark/`.
- Instructor Pilot export and annotation tooling source code under `src/batch_doc_vqa/datasets/`, as code only.
- Unit tests that construct synthetic temporary fixtures.
- Aggregate-only summaries that contain counts, hashes, metric values, timings, and costs, provided they include no raw student text, images, names, IDs, submission IDs, grades, comments, filenames derived from private IDs, or per-student examples.

## Private-Local Material

These must not be committed, pasted into chat, quoted in docs, copied into public reports, or uploaded to third-party systems except through an explicitly approved model-evaluation run:

- Instructor Pilot SQLite databases.
- Instructor Pilot media roots and submission images.
- Any local export images directory.
- `doc_info.csv` generated from real Instructor Pilot exports.
- `test_ids.csv` generated from real Instructor Pilot exports.
- `submission_metadata.csv`.
- `dataset_manifest.json`, `cohort_definition.json`, and `ta_dataset.yaml` when generated from real Instructor Pilot exports.
- Model `results.json` files from real student images.
- `proposals.json`, because it may include identity guesses.
- `annotations/`, including `annotations/docs/doc-<id>.json`.
- scored reports or comparisons when they include per-document details, examples, filenames, or raw prediction text.
- screenshots of the annotation UI showing student work.

Default private locations:

- `/tmp/instructor_pilot_ta_v1/`
- `/tmp/instructor_pilot_export_full/`
- `ta_artifacts/`
- `annotations/`
- `tests/output/ta_runs/`
- `tests/output/runs/`

## Assistant Handling Rules

Allowed:

- inspect code, schemas, prompts, tests, and documentation;
- run git boundary checks;
- run tests that create synthetic temporary fixtures;
- report aggregate counts and hash presence without showing rows;
- discuss command structure and expected artifact types.

Not allowed without a fresh, explicit user instruction:

- open files under `ta_artifacts/`, `annotations/`, `/tmp/instructor_pilot_ta_v1/`, or `/tmp/instructor_pilot_export_full/`;
- display rows from real `doc_info.csv`, `test_ids.csv`, `submission_metadata.csv`, manifests, proposals, labels, results, or reports;
- inspect exported images or UI screenshots from real student submissions;
- commit, stage, or move private-local artifacts;
- use raw student examples in public README, project cards, CV bullets, reports, or screenshots.

If a command accidentally reports private rows, stop and do not repeat or summarize those rows. Use aggregate-only commands next.

## Tier 1 Protocol S/M Pilot Checklist

This checklist is the concrete next path for the TA benchmark. It avoids exposing raw student artifacts.

### 0. Preflight Safety Boundary

Run from `/Users/ioannism/repos/batch-doc-vqa`:

```bash
git status --short
git check-ignore -v annotations/ tests/output/ta_runs/ ta_artifacts/
git status --ignored --short ta_artifacts annotations tests/output/ta_runs tests/output/runs
```

Pass condition:

- no private-local artifact is staged or tracked;
- expected local TA/output directories are ignored;
- any untracked public doc/code file is intentional.

### 1. Export Only In A Private Workspace

Use `instructor_pilot.py profile` first to inspect aggregate assignment counts only. Do not paste row-level output.

When exporting, write only to `/tmp/instructor_pilot_ta_v1/export/` or an ignored `ta_artifacts/.../export/` path. Use dry-run before `--apply`.

Pass condition:

- export target is outside tracked public files;
- no exported rows or images are displayed in the transcript;
- the export command records only aggregate counts in any public note.

### 2. Freeze Cohort Metadata

Run `dataset_freeze` against private-local export files and write outputs to the same private workspace.

Pass condition:

- `dataset_hash` and split counts are recorded;
- `tier1_docs`, `tier2_docs`, `dev_docs`, and `test_docs` exist structurally;
- no document rows, names, IDs, submissions, or image filenames are exposed.

### 3. Build Fixed 30-Doc Dev Subset

Run `split_subset` for `dev_docs`, following `docs/TA_TIER1_PROTOCOLS.md`.

Pass condition:

- dev subset files exist in the private workspace;
- both Protocol S and Protocol M use the exact same dev subset;
- only counts and paths are reported.

### 4. Run Protocol S And Protocol M

Run the two Tier 1 inference tracks from `docs/TA_TIER1_PROTOCOLS.md`:

- Protocol S: `ta-tier1-single-pass.md` plus `ta_tier1_single_pass.schema.json`.
- Protocol M Step A: `ta-tier1-page-extract.md` plus `ta_tier1_page_extract.schema.json`.

Pass condition:

- results are written only under the private workspace;
- provider and retention choices are acceptable for real student images before the run starts;
- stored result files do not persist provider generation IDs beyond the local recovery step, following `AGENTS.md`.

### 5. Convert Results To Private Proposals

Run `ta_benchmark.proposals` separately for each protocol.

Pass condition:

- proposal files stay private;
- no identity guesses, raw problem text, evidence IDs tied to filenames, or per-document predictions are pasted into public notes.

### 6. Human Review In Local UI

Launch `ta_annotation_ui` against the private dev subset and proposals.

Pass condition:

- labels are saved under private `annotations/docs/`;
- each reviewed label has reviewer status and proposal action;
- non-trivial problem labels cite evidence regions;
- UI screenshots are not shared unless fully anonymized and reviewed.

### 7. Score And Compare Aggregate Metrics

Run `score_runs`, `report`, `compare_runs`, and optionally `agreement` on private labels/results.

Pass condition:

- public discussion is limited to aggregate metrics, costs, runtimes, counts, and decision rationale;
- per-document examples remain private;
- any report intended for public use is reviewed for identifiers, filenames, raw text, images, submission IDs, grades, comments, and traces.

### 8. Freeze Protocol Decision

Choose Protocol S or M for broader Tier 1 labeling only after:

- the same 30-doc dev subset was used;
- human-reviewed labels are validated;
- quality, cost, runtime, and error-attribution tradeoffs are documented at aggregate level;
- no private artifact boundary violations occurred.

## Public Claim Boundary

Until the pilot is complete and a sanitized summary is written, the public claim should remain conservative:

> Implemented local-only TA benchmark tooling for student-submission document workflows, including schema validation, proposal generation, annotation UI support, scoring/reporting modules, and privacy boundaries for real student artifacts.

Do not claim a completed public TA benchmark, large-scale human-reviewed dataset, or deployable student-grading system until the Tier 1 pilot has completed under this safety process and the resulting public summary has been sanitized.
