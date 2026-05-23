# Default Student Synthetic Suite

This is the public-safe benchmark suite for the `default_student` task. It is intentionally separate from TA and Instructor Pilot artifacts: it uses only tracked synthetic entity rows in `docs/examples/synthetic/default_student_suite_v1_entities.csv` and writes generated outputs under the ignored `tests/output/synthetic/` tree.

## Versioned Suite

Suite spec:

- `docs/examples/synthetic/default_student_suite_v1.yaml`
- `suite_id`: `default_student_synthetic_v1`
- `task_type`: `default_student`
- Output root: `tests/output/synthetic/default_student_synthetic_v1`

Variants:

| Variant | Profile | Seed | Documents | Pages per doc | Target pages |
| --- | --- | ---: | ---: | ---: | --- |
| `clean` | `clean` | `4101` | 12 | 4 | 1, 3 |
| `tabular` | `tabular` | `4201` | 12 | 4 | 1, 3 |
| `noisy_mixed` | `noisy_mixed` | `4301` | 12 | 4 | 1, 3 |

Each variant writes:

- `task_docs.pdf`
- `test_ids.csv`
- `generation_plan.json`
- `metadata.json`
- `entities.normalized.json`
- `images/doc_info.csv`
- `BENCHMARKS.md`
- `pareto_plot.png`
- `pareto_plot_id_lev.png`

## Regenerate

```bash
uv run --with pymupdf generate-default-student-synthetic-suite --overwrite
```

To write the same suite somewhere else:

```bash
uv run --with pymupdf generate-default-student-synthetic-suite \
  --output-root /tmp/default_student_synthetic_v1 \
  --overwrite
```

To regenerate one variant:

```bash
uv run --with pymupdf generate-default-student-synthetic-suite \
  --variant noisy_mixed \
  --overwrite
```

The command does not read `annotations/`, `ta_artifacts/`, `/tmp/instructor_pilot_ta_v1`, or `/tmp/instructor_pilot_export_full`. It only reads the tracked suite spec and tracked synthetic entity CSV, then writes new generated artifacts under the selected output root.

## Run Models Against a Variant

After generation, run inference against one variant's images and manifest:

```bash
uv run openrouter-inference \
  --preset default_student \
  --images-dir tests/output/synthetic/default_student_synthetic_v1/noisy_mixed/images \
  --dataset-manifest tests/output/synthetic/default_student_synthetic_v1/noisy_mixed/images/doc_info.csv \
  --concurrency 64 \
  --rate-limit 64
```

Then rerun the suite command to refresh the benchmark table and Pareto plots for matching model runs:

```bash
uv run --with pymupdf generate-default-student-synthetic-suite \
  --variant noisy_mixed \
  --overwrite
```

If no matching model runs exist yet, the command writes explicit placeholder report artifacts instead of fabricating scores.
