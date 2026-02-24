# Comprehensive Benchmark Results

This document contains benchmark results for a specific batch document VQA scoring setup.

Scoring dataset:

- **Document manifest**: `imgs/q11/doc_info.csv`
- **Ground truth**: `tests/data/test_ids.csv`
- **Logical documents**: **32**

## Test Configuration

All VLM runs use the following consistent configuration:
- **# samples**: 1 (single inference per document)
- **Structured output**: JSON schema enforcement via OpenRouter structured outputs or outlines library
- **Target fields**: Determined by the selected extraction preset and scoring dataset
- **Regex pattern**: Yes for local models with outlines, No for API models with structured outputs

## Baseline Comparison

- **OpenCV+CNN**: Traditional computer vision pipeline with convolutional neural networks
  - **logits available**: Yes (provides confidence scores and top-k predictions)
  - **digit_top2**: 90.62%, **digit_top3**: 94.14% (multi-candidate accuracy)
  - **Model licensing**: N/A (uses OpenCV + custom CNN)

## Key Metrics

- **digit_top1**: Percentage of individual digits correctly identified in 8-digit IDs
- **8-digit id_top1**: Percentage of complete 8-digit IDs matched exactly
- **lastname_top1**: Percentage of last names matched exactly  
- **ID Avg d_Lev**: Average Levenshtein distance for ID matching (lower is better)
- **Lastname Avg d_Lev**: Average Levenshtein distance for lastname matching (lower is better)
- **Docs detected**: Percentage of documents successfully processed
- **Runtime**: Processing time for the entire dataset
- **Cost per image**: Average cost per document image processed
- **Total cost**: Total cost for processing all scored documents

**Bold** values indicate the best performance for each metric.

## Complete Results

| **Metric** | **OpenCV+CNN** | **qwen**<br>qwen3.5-plus-02-15 | **qwen**<br>qwen2.5-vl-72b-instruct | **qwen**<br>qwen-vl-plus | **qwen**<br>qwen3-vl-8b-instruct | **qwen**<br>qwen3-vl-32b-instruct | **qwen**<br>qwen3-vl-8b-thinking | **qwen**<br>qwen3-vl-30b-a3b-instruct | **qwen**<br>qwen3-vl-30b-a3b-thinking | **qwen**<br>qwen-vl-max | **qwen**<br>qwen3-vl-235b-a22b-thinking | **qwen**<br>qwen3-vl-235b-a22b-instruct | **qwen**<br>qwen2.5-vl-32b-instruct | **baidu**<br>ernie-4.5-vl-424b-a47b | **google**<br>gemma-3-4b-it | **google**<br>gemini-2.5-flash-lite-preview-09-2025 | **google**<br>gemini-2.5-flash-lite | **google**<br>gemma-3-27b-it | **google**<br>gemma-3-12b-it | **google**<br>gemini-3-flash-preview | **google**<br>gemini-3-pro-preview | **google**<br>gemini-2.5-flash-preview-09-2025 | **google**<br>gemini-2.5-pro | **google**<br>gemini-2.5-flash | **amazon**<br>nova-lite-v1 | **amazon**<br>nova-2-lite-v1 | **amazon**<br>nova-premier-v1 | **moonshotai**<br>kimi-k2.5 | **moonshotai**<br>kimi-vl-a3b-thinking:free | **bytedance-seed**<br>seed-1.6 | **bytedance-seed**<br>seed-1.6-flash | **mistralai**<br>ministral-8b-2512 | **mistralai**<br>ministral-3b-2512 | **mistralai**<br>mistral-large-2512 | **openai**<br>gpt-5.2-chat | **openai**<br>gpt-5.2 | **openai**<br>gpt-5.1-chat | **openai**<br>gpt-5.1 | **openai**<br>gpt-5-chat | **openai**<br>gpt-5-mini | **openai**<br>gpt-5-nano | **z-ai**<br>glm-4.6v | **z-ai**<br>glm-4.5v | **anthropic**<br>claude-haiku-4.5 | **anthropic**<br>claude-sonnet-4.5 | **anthropic**<br>claude-sonnet-4 | **microsoft**<br>phi-4-multimodal-instruct | **bytedance**<br>ui-tars-1.5-7b | **meta-llama**<br>llama-4-maverick |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| LLM model size | N/A | ?? | 72B | ?? | 8B | 32B | 8B | 30A3 | 30A3 | ?? | 235A22 | 235A22 | 32B | 424A47 | 4B | ?? | ?? | 27B | 12B | ?? | ?? | ?? | ?? | ?? | ?? | ?? | ?? | 1000A32 | 16A3 | ?? | ?? | 8B | 3B | 675A41 | ?? | ?? | ?? | ?? | ?? | ?? | ?? | 106B | 106A12 | ?? | ?? | ?? | 5.6B | 7B | 400A17 |
| Open-weights | N/A | No | Yes | No | Yes | Yes | Yes | Yes | Yes | No | Yes | Yes | Yes | Yes | Yes | No | No | Yes | Yes | No | No | No | No | No | No | No | No | Yes | Yes | No | No | Yes | Yes | Yes | No | No | No | No | No | No | No | Yes | Yes | No | No | No | Yes | Yes | Yes |
| digit_top1 | 85.16% | 98.83% | 96.48% | 96.88% | 99.61% | 97.27% | 92.19% | 99.22% | 97.27% | 97.27% | 94.53% | 97.66% | 96.09% | 98.05% | 72.27% | 98.05% | 99.22% | 84.18% (n=2) | 75.39% | 99.22% | 99.22% | 98.05% | 99.22% | 98.83% | 93.36% | 97.27% | 94.53% | **100.00%** | 85.94% | 96.48% | 95.70% | 89.84% | 83.59% | 78.12% | 98.44% | 98.44% | 91.80% | 92.58% | 89.84% | 98.83% | 96.48% | 94.92% | 93.36% | 74.22% | 82.81% | 84.77% | 71.48% | 96.48% | 89.84% |
| 8-digit id_top1 | ?? | 90.62% | 84.38% | 75.00% | 96.88% | 84.38% | 71.88% | 93.75% | 81.25% | 81.25% | 78.12% | 84.38% | 84.38% | 87.50% | 43.75% | 90.62% | 93.75% | 56.25% (n=2) | 43.75% | 93.75% | 93.75% | 84.38% | 93.75% | 90.62% | 75.00% | 78.12% | 75.00% | **100.00%** | 50.00% | 78.12% | 78.12% | 71.88% | 53.12% | 37.50% | 90.62% | 90.62% | 68.75% | 71.88% | 62.50% | 90.62% | 78.12% | 81.25% | 78.12% | 21.88% | 40.62% | 37.50% | 40.62% | 84.38% | 56.25% |
| lastname_top1 | N/A | 96.88% | **100.00%** | **100.00%** | **100.00%** | 96.88% | 90.62% | 93.75% | 96.88% | 96.88% | **100.00%** | 96.88% | **100.00%** | 93.75% | 93.75% | **100.00%** | 96.88% | **100.00% (n=2)** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 87.50% | **100.00%** | **100.00%** | **100.00%** | 96.88% | 93.75% | 96.88% | 93.75% | 90.62% | 87.50% | 96.88% | 96.88% | **100.00%** | **100.00%** | **100.00%** | **100.00%** | 93.75% | **100.00%** | **100.00%** | 93.75% | 96.88% | **100.00%** | **100.00%** | 96.88% | 96.88% |
| ID Avg d_Lev | N/A | 0.0938 | 0.1562 | 0.2500 | 0.0312 | 0.2188 | 0.5000 | 0.0625 | 0.2188 | 0.2188 | 0.4688 | 0.1875 | 0.1562 | 0.1562 | 1.1875 | 0.0938 | 0.0625 | 0.9375 (n=2) | 0.9688 | 0.0625 | 0.0625 | 0.1562 | 0.0625 | 0.0938 | 0.3125 | 0.2188 | 0.2500 | **0.0000** | 0.9062 | 0.2812 | 0.2812 | 0.4062 | 0.5312 | 1.2188 | 0.1250 | 0.1250 | 0.4375 | 0.5938 | 0.5312 | 0.0938 | 0.2188 | 0.1875 | 0.2188 | 1.4062 | 1.0000 | 1.0938 | 1.2188 | 0.1562 | 0.5312 |
| Lastname Avg d_Lev | N/A | 0.0312 | **0.0000** | **0.0000** | **0.0000** | 0.0312 | 0.2812 | 0.1875 | 0.0312 | 0.0312 | **0.0000** | 0.0312 | **0.0000** | 0.0938 | 0.0938 | **0.0000** | 0.0312 | **0.0000 (n=2)** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 0.2500 | **0.0000** | **0.0000** | **0.0000** | 0.0312 | 0.1875 | 0.0312 | 0.0938 | 0.2188 | 0.3750 | 0.0625 | 0.1562 | **0.0000** | **0.0000** | **0.0000** | **0.0000** | 0.0625 | **0.0000** | **0.0000** | 0.2188 | 0.0625 | **0.0000** | **0.0000** | 0.0312 | 0.0312 |
| Docs detected | 90.62% (29/32) | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32) (n=2)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** |
| Runtime | **~1 second** | 4.0 minutes | 19 seconds | 10 seconds | 12 seconds | 2.6 minutes | 14.4 minutes | 2.3 minutes | 6.3 minutes | 4.5 minutes | 13.9 minutes | 4.5 minutes | 2.3 minutes | 51 seconds | 14 seconds | 11 seconds | 11 seconds | 17 seconds (n=2) | 10 seconds | 8 seconds | 26.4 minutes | 4.4 minutes | 8.0 minutes | 2.7 minutes | 14 seconds | 3.5 minutes | 6.4 minutes | 1.6 minutes | 9.5 minutes | 57 seconds | 44 seconds | 14 seconds | 18 seconds | 23 seconds | 53 seconds | 57 seconds | 5.2 minutes | 8.9 minutes | 2.8 minutes | 8.3 minutes | 10.5 minutes | 6.6 minutes | 6.2 minutes | 5.0 minutes | 5.4 minutes | 3.5 minutes | 2.1 minutes | 1.8 minutes | 2.0 minutes |
| Cost per image | **$0.00** | $0.003124 | $0.001033 | $0.000314 | $0.000266 | $0.000814 | $0.001563 | $0.000700 | $0.001311 | $0.001415 | $0.002376 | $0.000786 | $0.002605 | $0.000703 | $0.000021 | $0.000151 | $0.000214 | $0.000054 (n=2) | $0.000040 | $0.001636 | $0.015111 | $0.000418 | $0.007125 | $0.000695 | $0.000115 | $0.000911 | $0.005061 | $0.004679 | **$0.000000** | $0.000770 | $0.000196 | $0.000388 | $0.000265 | $0.001335 | $0.005232 | $0.005352 | $0.001866 | $0.002870 | $0.001260 | $0.001115 | $0.000463 | $0.001004 | $0.002057 | $0.001882 | $0.005646 | $0.005567 | $0.000025 | $0.000293 | $0.000539 |
| Total cost | **$0.00** | $0.1999 | $0.0661 | $0.0201 | $0.0171 | $0.0521 | $0.1000 | $0.0448 | $0.0839 | $0.0906 | $0.1521 | $0.0503 | $0.1667 | $0.0450 | $0.0013 | $0.0096 | $0.0137 | $0.0034 (n=2) | $0.0026 | $0.1047 | $0.9671 | $0.0267 | $0.4560 | $0.0445 | $0.0073 | $0.0583 | $0.3239 | $0.2995 | **$0.0000** | $0.0492 | $0.0126 | $0.0248 | $0.0170 | $0.0854 | $0.3348 | $0.3426 | $0.1195 | $0.1837 | $0.0807 | $0.0714 | $0.0297 | $0.0642 | $0.1316 | $0.1205 | $0.3614 | $0.3563 | $0.0016 | $0.0187 | $0.0345 |