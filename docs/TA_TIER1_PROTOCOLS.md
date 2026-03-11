# TA Tier 1 Protocol Pilot (30 Docs)

This runbook defines two Tier 1 inference tracks for pilot comparison:

- Protocol S: single-pass prompt/schema
- Protocol M: decomposed page extraction prompt/schema (Step A)

Both tracks should be run on the same frozen dataset and compared on quality + cost/runtime.

## Inputs

- images: `/tmp/instructor_pilot_ta_v1/export/images`
- doc manifest: `/tmp/instructor_pilot_ta_v1/export/doc_info.csv`
- test ids: `/tmp/instructor_pilot_ta_v1/export/test_ids.csv`

## Build 30-Doc Pilot Subset (dev_docs)

```bash
uv run python -m batch_doc_vqa.ta_benchmark.split_subset \
  --ta-dataset /tmp/instructor_pilot_ta_v1/ta_dataset.yaml \
  --split dev_docs \
  --output-dir /tmp/instructor_pilot_ta_v1/subsets
```

This creates:
- `/tmp/instructor_pilot_ta_v1/subsets/doc_info_dev_docs.csv`
- `/tmp/instructor_pilot_ta_v1/subsets/test_ids_dev_docs.csv`

## Protocol S (Single Pass)

```bash
uv run openrouter-inference \
  --model qwen/qwen3-vl-8b-instruct \
  --images-dir /tmp/instructor_pilot_ta_v1/export/images \
  --dataset-manifest /tmp/instructor_pilot_ta_v1/subsets/doc_info_dev_docs.csv \
  --pages 1,2,3,4 \
  --prompt-file docs/examples/prompts/ta-tier1-single-pass.md \
  --schema-file docs/examples/schemas/ta_tier1_single_pass.schema.json \
  --output-json /tmp/instructor_pilot_ta_v1/runs/protocol_s/results.json
```

## Protocol M (Decomposed, Step A)

```bash
uv run openrouter-inference \
  --model qwen/qwen3-vl-8b-instruct \
  --images-dir /tmp/instructor_pilot_ta_v1/export/images \
  --dataset-manifest /tmp/instructor_pilot_ta_v1/subsets/doc_info_dev_docs.csv \
  --pages 1,2,3,4 \
  --prompt-file docs/examples/prompts/ta-tier1-page-extract.md \
  --schema-file docs/examples/schemas/ta_tier1_page_extract.schema.json \
  --output-json /tmp/instructor_pilot_ta_v1/runs/protocol_m_step_a/results.json
```

## Convert Inference Results to Proposals

Use the same conversion command for either protocol:

```bash
uv run python -m batch_doc_vqa.ta_benchmark.proposals \
  --results-json /tmp/instructor_pilot_ta_v1/runs/<protocol>/results.json \
  --doc-info /tmp/instructor_pilot_ta_v1/subsets/doc_info_dev_docs.csv \
  --test-ids /tmp/instructor_pilot_ta_v1/subsets/test_ids_dev_docs.csv \
  --output /tmp/instructor_pilot_ta_v1/runs/<protocol>/proposals.json
```

## Notes

- For the 30-doc pilot, subset selection should be fixed and reused for both protocols.
- `ta_benchmark.proposals` now normalizes missing `page` / `evidence_id` / `problem_uid` fields when possible.
- Template matching (Protocol M Step C) remains doc-level and can be completed during annotation for this pilot.
