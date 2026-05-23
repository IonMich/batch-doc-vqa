# OpenRouter Model Notes

This file records notable operational observations for models that were attempted but did not produce a benchmark-grade run.

## qwen/qwen3.5-plus-20260420

- Attempt date: April 28, 2026
- Status: attempted, not retained as a benchmark artifact
- Outcome: `48/64` images succeeded, `16/64` failed
- Failure mode: provider-side `429 Provider returned error` responses on the routed `Alibaba` path
- Run profile: `temperature=0.6`, `top_p=0.95`, `top_k=20`, `min_p=0.0`, `presence_penalty=0.0`, `repetition_penalty=1.0`
- Notes:
  - This was not a parameter-mismatch run; it used the intended Qwen 3.5 Plus profile.
  - The issue was operational stability under batch load, not a scoring failure on completed responses.
  - The incomplete run directory was deleted so it would not be treated as a valid benchmark result.
  - If retried, use lower concurrency and treat the result as benchmark-eligible only if it completes cleanly.
