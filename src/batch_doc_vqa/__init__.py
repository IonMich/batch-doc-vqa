"""Batch Document VQA package."""

def main() -> None:
    """Main entry point for batch-doc-vqa CLI."""
    print("Batch Document VQA - Available commands:")
    print("")
    print("🤖 OpenRouter Inference:")
    print("  uv run openrouter-inference --model MODEL_NAME")
    print("  uv run openrouter-cli --model MODEL_NAME")
    print("  uv run openrouter-ui  # Interactive model selection")
    print("")
    print("📊 Benchmarks & Analysis:")
    print("  uv run update-benchmarks")
    print("  uv run generate-benchmark-table")
    print("  uv run generate-pareto-plot")
    print("  uv run update-readme-section")
    print("")
    print("💡 Examples:")
    print("  uv run openrouter-inference --model z-ai/glm-4.5v")
    print("  uv run openrouter-cli --interactive")
    print("  uv run update-benchmarks --patterns glm qwen")
    print("  uv run generate-benchmark-table --patterns anthropic openai")
    print("  uv run generate-pareto-plot --output pareto.png")
