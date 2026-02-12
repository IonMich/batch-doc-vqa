# Statistical Calibration Investigation (Legacy)

This document preserves the legacy statistical calibration experiment that produced the calibration artifacts referenced in this repository.

The goal of that experiment was calibration analysis (not model serving stack choice), so the details are kept here as historical context while the main README focuses on current OpenRouter workflows.

Full narrative and broader discussion:

- https://github.com/IonMich/batch-doc-vqa/wiki/Row-of-Digits-OCR:-OpenCV-CNN-versus-LLMs

Calibration artifact referenced from this repo:

- `tests/output/public/calibration_curves.png`

## Legacy Runtime Used in That Experiment

The original calibration run path used Ollama + `llamavision.py`.

### Setup

1. Install Ollama via the official [installation instructions](https://ollama.com/).
2. Pull a vision model, for example:

```bash
ollama pull llama3.2-vision
```

3. Install the Python wrapper in an isolated environment:

```bash
conda create -n ollama python=3.11
conda activate ollama
pip install ollama
```

4. Clone this repository:

```bash
git clone https://github.com/IonMich/batch-doc-vqa
cd batch-doc-vqa
```

5. Start the Ollama server (`ollama server` on Linux; app launch on macOS).

### Calibration-style Sampling Command

```bash
python llamavision.py --filepath imgs/sub-page-3.png --n_trials 10
```

The key parameter for repeated sampling was:

- `--n_trials`: number of repeated runs per image to estimate empirical output frequencies.

### Legacy CLI Reference

```bash
python llamavision.py [-h] [--filepath FILEPATH] [--pattern PATTERN] [--n_trials N_TRIALS] [--system SYSTEM] [--model MODEL] [--no-stream] [--top_k TOP_K]
```

Important options used in that setup:

- `--filepath`: file or directory of images to process.
- `--pattern`: filename filter when `--filepath` is a directory.
- `--n_trials`: repeated runs per image.
- `--system`: prompt selector index (`SYSTEM_PROMPTS` in `llamavision.py`).
- `--model`: vision model name.
- `--top_k`: output distribution breadth.
- `--no-stream`: disable token streaming.

## Notes

- This is historical/legacy experiment documentation.
- Current primary extraction and benchmarking workflows are documented in `README.md`.
- If you want this migrated to the GitHub wiki page itself, copy this file into the corresponding wiki article.
