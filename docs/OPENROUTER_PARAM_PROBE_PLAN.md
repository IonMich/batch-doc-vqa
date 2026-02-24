# OpenRouter Parameter-Respect Probe Plan (Draft)

## Status
This is a high-level draft only.
Implementation details and thresholds should be discussed before coding.

## Goal
Build a robust way to test whether requested model parameters are actually respected by the routed provider/model path.

## Proposed V1 Scope
1. Add a probe command:
   - `uv run openrouter-param-probe --model <model_id> ...`
2. Run controlled A/B parameter sweeps with repeated trials.
3. Use parameter-specific probe prompts and analyze output behavior statistically.
4. Classify each parameter as:
   - `respected`
   - `weak_effect`
   - `likely_ignored`
5. Save artifacts:
   - `raw_responses.jsonl`
   - `analysis.json`
   - `summary.md`

## Parameters To Probe
1. Sampling controls:
   - `temperature`
   - `top_p`
   - `top_k`
   - `min_p`
2. Penalties:
   - `repetition_penalty`
   - `presence_penalty`

## Measurement Strategy
1. For sampling controls:
   - Measure diversity/entropy sensitivity across repeated generations.
2. For repetition penalty:
   - Measure repetition score deltas on long continuation probes.
3. For presence penalty:
   - Measure novelty rate increase in generated tokens/topics.

## Reliability Controls
1. Stabilize routing:
   - fixed provider order
   - no fallbacks
   - low concurrency
2. Use effect-size + confidence thresholds (not single-run anecdotes).
3. Require consistency across multiple prompts per parameter.

## Open Discussion Items (Must Be Decided First)
1. Exact probe prompt set and domains.
2. Trial counts per condition.
3. Statistical tests and minimum effect thresholds.
4. Pass/fail definitions and CI policy.
5. How to handle provider route drift during a probe run.

