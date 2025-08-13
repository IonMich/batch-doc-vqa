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


| Metric | OpenCV+CNN | meta-llama/llama-4-maverick | moonshotai/kimi-vl-a3b-thinking:free | anthropic/claude-sonnet-4 | openai/gpt-5-nano | z-ai/glm-4.5v | qwen/qwen-2.5-vl-7b-instruct | qwen/qwen2.5-vl-32b-instruct |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| LLM model size | N/A | 400A17 | 16A3 | ?? | ?? | 106A12 | 7B | 32B |
| Open-weights | N/A | Yes | Yes | No | No | Yes | Yes | Yes |
| digit_top1 | 85.16% | 89.84% | 85.94% | 84.77% | **96.48%** | 93.36% | 82.08% | 96.09% |
| 8-digit id_top1 | ?? | 56.25% | 50.00% | 37.50% | 78.12% | 78.12% | 76.67% | **84.38%** |
| lastname_top1 | N/A | 93.75% | 96.88% | **100.00%** | 90.62% | **100.00%** | **100.00%** | **100.00%** |
| ID Avg d_Lev | N/A | 0.5312 | 0.9062 | 1.0938 | 0.2188 | 0.2188 | 1.6333 | **0.1562** |
| Lastname Avg d_Lev | N/A | 0.0938 | 0.0938 | **0.0000** | 0.1250 | **0.0000** | **0.0000** | **0.0000** |
| Docs detected | 90.62% (29/32) | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | **100.00% (32/32)** | 93.75% (30/32) | **100.00% (32/32)** |
| Runtime | **~ 1 second** | 2.0 minutes | 9.5 minutes | 3.5 minutes | 10.5 minutes | 6.2 minutes | 6.8 minutes | 2.3 minutes |
| Cost per image | **$0.00** | $0.000539 | **$0.000000** | $0.005567 | $0.000463 | $0.002057 | $0.000081 | $0.002605 |
| Total cost | **$0.00** | $0.0345 | **$0.0000** | $0.3563 | $0.0297 | $0.1316 | $0.0039 | $0.1667 |