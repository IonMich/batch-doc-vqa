# Comprehensive Benchmark Results

This document contains benchmark results for a specific batch document VQA scoring setup.

Scoring dataset:

- **Document manifest**: `{{DOC_INFO_FILE}}`
- **Ground truth**: `{{TEST_IDS_FILE}}`
- **Logical documents**: **{{EXPECTED_DOCS}}**

## Test Configuration

All VLM runs use the following consistent configuration:
- **# samples**: 1 (single inference per document)
- **Structured output**: JSON schema enforcement via OpenRouter structured outputs or outlines library
- **Target fields**: Determined by the selected extraction preset and scoring dataset
- **Regex pattern**: Yes for local models with outlines, No for API models with structured outputs

{{BASELINE_SECTION}}

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
