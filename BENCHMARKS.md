# Comprehensive Benchmark Results

This document contains the complete benchmark results for all tested models on the batch document VQA task. The test dataset consists of 32 documents (Physics quiz sheets) that need to be matched to students via their 8-digit university ID and last names.

## Test Configuration

All VLM runs use the following consistent configuration:
- **# samples**: 1 (single inference per document)
- **Structured output**: JSON schema enforcement via OpenRouter structured outputs or outlines library
- **Detect Type**: LastName (2 pages) + ID (1 page) for VLMs vs ID (1 page) only for OpenCV+CNN
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
- **Total cost**: Total cost for processing all 32 documents

**Bold** values indicate the best performance for each metric.

## Complete Results


| **Metric** | **OpenCV+CNN** | **qwen**<br>qwen3-vl-30b-a3b-instruct | **qwen**<br>qwen3-vl-30b-a3b-thinking | **qwen**<br>qwen-vl-max | **qwen**<br>qwen3-vl-235b-a22b-thinking | **qwen**<br>qwen3-vl-235b-a22b-instruct | **qwen**<br>qwen-2.5-vl-7b-instruct | **qwen**<br>qwen2.5-vl-32b-instruct | **google**<br>gemini-2.5-flash-preview-09-2025 | **google**<br>gemini-2.5-flash-lite-preview-09-2025 | **google**<br>gemini-2.5-pro | **google**<br>gemini-2.5-flash-lite | **google**<br>gemma-3-27b-it | **google**<br>gemma-3-4b-it | **google**<br>gemini-2.5-flash | **anthropic**<br>claude-sonnet-4.5 | **anthropic**<br>claude-sonnet-4 | **openai**<br>gpt-5-chat | **openai**<br>gpt-5-mini | **openai**<br>gpt-5-nano | **amazon**<br>nova-lite-v1 | **microsoft**<br>phi-4-multimodal-instruct | **bytedance**<br>ui-tars-1.5-7b | **meta-llama**<br>llama-4-maverick | **moonshotai**<br>kimi-vl-a3b-thinking:free | **z-ai**<br>glm-4.5v |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| LLM model size | N/A | 30A3 | 30A3 | ?? | 235A22 | 235A22 | 7B | 32B | ?? | ?? | ?? | ?? | 27B | 4B | ?? | ?? | ?? | ?? | ?? | ?? | ?? | 5.6B | 7B | 400A17 | 16A3 | 106A12 |
| Open-weights | N/A | Yes | Yes | No | Yes | Yes | Yes | Yes | No | No | No | No | Yes | Yes | No | No | No | No | No | No | No | Yes | Yes | Yes | Yes | Yes |
| digit_top1 | 85.16% | **99.22%** | 97.27% | 97.27% | 94.53% | 97.66% | 82.08% | 96.09% | 98.05% | 97.66% | **99.22%** | **99.22%** | 89.45% | 75.00% | 98.83% | 82.81% | 84.77% | 89.84% | 98.83% | 96.48% | 89.06% | 71.48% | 96.48% | 89.84% | 85.94% | 93.36% |
| 8-digit id_top1 | ?? | **93.75%** | 81.25% | 81.25% | 78.12% | 84.38% | 76.67% | 84.38% | 84.38% | 84.38% | **93.75%** | **93.75%** | 65.62% | 40.62% | 90.62% | 40.62% | 37.50% | 62.50% | 90.62% | 78.12% | 75.00% | 40.62% | 84.38% | 56.25% | 50.00% | 78.12% |
| lastname_top1 | N/A | 93.75% | 93.75% | 96.88% | **100.00%** | 96.88% | **100.00%** | **100.00%** | 96.88% | 96.88% | 96.88% | 93.75% | **100.00%** | 90.62% | 96.88% | 93.75% | **100.00%** | **100.00%** | 96.88% | 90.62% | 96.88% | **100.00%** | 96.88% | 93.75% | 96.88% | **100.00%** |
| ID Avg d_Lev | N/A | **0.0625** | 0.2188 | 0.2188 | 0.4688 | 0.1875 | 1.6333 | 0.1562 | 0.1562 | 0.1875 | **0.0625** | **0.0625** | 0.5000 | 0.9688 | 0.0938 | 1.0000 | 1.0938 | 0.5312 | 0.0938 | 0.2188 | 0.3750 | 1.2188 | 0.1562 | 0.5312 | 0.9062 | 0.2188 |
| Lastname Avg d_Lev | N/A | 0.1875 | 0.0625 | 0.0312 | **0.0000** | 0.0312 | **0.0000** | **0.0000** | 0.0312 | 0.0312 | 0.0312 | 0.0625 | **0.0000** | 0.1250 | 0.0312 | 0.0938 | **0.0000** | **0.0000** | 0.0312 | 0.1250 | 0.0312 | **0.0000** | 0.0312 | 0.0938 | 0.0938 | **0.0000** |
| Docs detected | 90.62% (29/32) | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | 93.75% (30/32) | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** |
| Runtime | **~1 second** | 2.3 minutes | 6.3 minutes | 4.5 minutes | 13.9 minutes | 4.5 minutes | 6.8 minutes | 2.3 minutes | 4.4 minutes | 4.4 minutes | 8.0 minutes | 2.3 minutes | 2.6 minutes | 2.0 minutes | 2.7 minutes | 5.4 minutes | 3.5 minutes | 2.8 minutes | 8.3 minutes | 10.5 minutes | 1.7 minutes | 2.1 minutes | 1.8 minutes | 2.0 minutes | 9.5 minutes | 6.2 minutes |
| Cost per image | **$0.00** | $0.000700 | $0.001311 | $0.001415 | $0.002376 | $0.000786 | $0.000081 | $0.002605 | $0.000418 | $0.000117 | $0.007125 | $0.000214 | $0.000054 | $0.000010 | $0.000695 | $0.005646 | $0.005567 | $0.001260 | $0.001115 | $0.000463 | $0.000114 | $0.000025 | $0.000293 | $0.000539 | **$0.000000** | $0.002057 |
| Total cost | **$0.00** | $0.0448 | $0.0839 | $0.0906 | $0.1521 | $0.0503 | $0.0039 | $0.1667 | $0.0267 | $0.0075 | $0.4560 | $0.0137 | $0.0035 | $0.0007 | $0.0445 | $0.3614 | $0.3563 | $0.0807 | $0.0714 | $0.0297 | $0.0073 | $0.0016 | $0.0187 | $0.0345 | **$0.0000** | $0.1316 |