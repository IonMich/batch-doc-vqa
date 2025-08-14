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


| **Metric** | **OpenCV+CNN** | **bytedance**<br>ui-tars-1.5-7b | **google**<br>gemini-2.5-flash-lite | **google**<br>gemma-3-27b-it | **google**<br>gemma-3-4b-it | **google**<br>gemini-2.5-flash | **openai**<br>gpt-5-mini | **openai**<br>gpt-5-nano | **meta-llama**<br>llama-4-maverick | **moonshotai**<br>kimi-vl-a3b-thinking:free | **anthropic**<br>claude-sonnet-4 | **z-ai**<br>glm-4.5v | **qwen**<br>qwen-2.5-vl-7b-instruct | **qwen**<br>qwen2.5-vl-32b-instruct |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| LLM model size | N/A | 7B | ?? | 27B | 4B | ?? | ?? | ?? | 400A17 | 16A3 | ?? | 106A12 | 7B | 32B |
| Open-weights | N/A | Yes | No | Yes | Yes | No | No | No | Yes | Yes | No | Yes | Yes | Yes |
| digit_top1 | 85.16% | 96.48% | **99.22%** | 89.45% | 75.00% | 98.83% | 98.83% | 96.48% | 89.84% | 85.94% | 84.77% | 93.36% | 82.08% | 96.09% |
| 8-digit id_top1 | ?? | 84.38% | **93.75%** | 65.62% | 40.62% | 90.62% | 90.62% | 78.12% | 56.25% | 50.00% | 37.50% | 78.12% | 76.67% | 84.38% |
| lastname_top1 | N/A | 96.88% | 93.75% | **100.00%** | 90.62% | 96.88% | 96.88% | 90.62% | 93.75% | 96.88% | **100.00%** | **100.00%** | **100.00%** | **100.00%** |
| ID Avg d_Lev | N/A | 0.1562 | **0.0625** | 0.5000 | 0.9688 | 0.0938 | 0.0938 | 0.2188 | 0.5312 | 0.9062 | 1.0938 | 0.2188 | 1.6333 | 0.1562 |
| Lastname Avg d_Lev | N/A | 0.0312 | 0.0625 | **0.0000** | 0.1250 | 0.0312 | 0.0312 | 0.1250 | 0.0938 | 0.0938 | **0.0000** | **0.0000** | **0.0000** | **0.0000** |
| Docs detected | 90.62% (29/32) | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | 93.75% (30/32) | **100.00% (32/32)** |
| Runtime | **~ 1 second** | 1.8 minutes | 2.3 minutes | 2.6 minutes | 2.0 minutes | 2.7 minutes | 8.3 minutes | 10.5 minutes | 2.0 minutes | 9.5 minutes | 3.5 minutes | 6.2 minutes | 6.8 minutes | 2.3 minutes |
| Cost per image | **$0.00** | $0.000293 | $0.000214 | $0.000054 | $0.000010 | $0.000695 | $0.001115 | $0.000463 | $0.000539 | **$0.000000** | $0.005567 | $0.002057 | $0.000081 | $0.002605 |
| Total cost | **$0.00** | $0.0187 | $0.0137 | $0.0035 | $0.0007 | $0.0445 | $0.0714 | $0.0297 | $0.0345 | **$0.0000** | $0.3563 | $0.1316 | $0.0039 | $0.1667 |