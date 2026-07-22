# Q11 New Model Benchmark Runbook

Use this runbook when refreshing the default q11 benchmark for newly available vision models.

## Scope

- Always use the default q11 benchmark dataset.
- Default PDF: `imgs/quiz11-presidents.pdf`
- Default images directory: `imgs/q11`
- Default document manifest: `imgs/q11/doc_info.csv`
- Default ground truth: `tests/data/test_ids.csv`
- Default preset: `default_student`
- Default benchmark pages: `1,3`

## Preflight

1. Check the worktree before any benchmark run:

```bash
git status --short
```

2. If there are reproducibility-relevant changes, inspect them and commit them when they are coherent and testable. Do not benchmark from a dirty relevant worktree unless the operator explicitly accepts a non-comparable run.

3. Confirm q11 images and manifest exist. Regenerate if missing or stale:

```bash
uv run --with pymupdf pdf-to-imgs \
  --filepath imgs/quiz11-presidents.pdf \
  --pages_i 4 \
  --dpi 300 \
  --output_dir imgs/q11
```

4. Before proposing a new model run, spend a few minutes checking current model metadata:

- OpenRouter model id and image-input support.
- OpenRouter supported parameters, especially `response_format` or `structured_outputs`.
- Official model card or Hugging Face card for model size, open-weight status, and license.
- Whether the model is already present in `BENCHMARKS.md` or `tests/output/runs/`.

## Protected Run Command

Use fully parallel q11 inference by default. q11 has 32 logical documents and pages `1,3`, so the default run processes 64 images.

```bash
uv run openrouter-inference \
  --preset default_student \
  --model <openrouter_model_id> \
  --images-dir imgs/q11 \
  --dataset-manifest imgs/q11/doc_info.csv \
  --pages 1,3 \
  --concurrency 64 \
  --rate-limit 64 \
  --provider-data-collection deny \
  --provider-zdr
```

For known open-weight models, include explicit metadata:

```bash
uv run openrouter-inference \
  --preset default_student \
  --model <openrouter_model_id> \
  --model-size <size> \
  --open-weights \
  --license-info <license> \
  --images-dir imgs/q11 \
  --dataset-manifest imgs/q11/doc_info.csv \
  --pages 1,3 \
  --concurrency 64 \
  --rate-limit 64 \
  --provider-data-collection deny \
  --provider-zdr
```

## Privacy Fallbacks

Keep privacy routing protected unless that route fails.

1. First choice:

```bash
--provider-data-collection deny --provider-zdr
```

2. If no ZDR route exists, retry without ZDR but still deny data collection:

```bash
--provider-data-collection deny --no-provider-zdr
```

3. Only if that still fails and the operator explicitly accepts the tradeoff, allow data collection:

```bash
--provider-data-collection allow --no-provider-zdr
```

Do not use `--skip-reproducibility-checks` for clean benchmark runs.

## Refresh Benchmark Artifacts

After successful inference runs, publish the sanitized benchmark evidence before
regenerating public artifacts:

```bash
uv run publish-benchmark-runs --patterns '<new-run-name-regex>' --finalize
uv run update-benchmarks --source published --no-interactive
uv run update-benchmarks --source published --no-interactive --check
```

Raw `tests/output/runs/` artifacts remain ignored and machine-local. The
published archive retains run configuration, aggregation provenance, aggregate
scores, exact request scope, and aggregate timing/token/cost/failure
evidence—never raw names, IDs, prompts, paths, per-document references, or
provider request identifiers. Publication is strict by default: every selected
run must exactly cover the recorded pages and pass all archive invariants.

Dataset identity uses the canonical document/page structure, test targets, and
source PDF hash, so harmless randomized image filename suffixes do not split one
logical dataset across machines. Published-only regeneration reads that identity
from the finalized archive and does not require local scoring inputs or raw runs.

Keep raw runs in private storage if future work needs to rescore the extracted
text itself. The public archive intentionally cannot reconstruct those values.

If unknown model metadata is detected, review/update `model_metadata.json` with:

- `open_weights`
- `model_size`
- `license`

Then rerun:

```bash
uv run update-benchmarks
```

## Current Candidate Commands

Mistral Medium 3.5:

```bash
uv run openrouter-inference \
  --preset default_student \
  --model mistralai/mistral-medium-3-5 \
  --model-size 128B \
  --open-weights \
  --license-info "Modified MIT" \
  --images-dir imgs/q11 \
  --dataset-manifest imgs/q11/doc_info.csv \
  --pages 1,3 \
  --concurrency 64 \
  --rate-limit 64 \
  --provider-data-collection deny \
  --provider-zdr
```

Gemini 3.5 Flash:

```bash
uv run openrouter-inference \
  --preset default_student \
  --model google/gemini-3.5-flash \
  --images-dir imgs/q11 \
  --dataset-manifest imgs/q11/doc_info.csv \
  --pages 1,3 \
  --concurrency 64 \
  --rate-limit 64 \
  --provider-data-collection deny \
  --provider-zdr
```

## Final Report

Report:

- Exact commands run.
- Whether privacy routing stayed protected or was relaxed.
- Run names and artifact paths.
- Benchmark rank/delta, cost, runtime, and failures.
- Files changed and whether the repo is clean afterward.
