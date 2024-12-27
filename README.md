# Vision LLM - Batch Document VQA with structured responses

![Probability calibration curves for OpenCV+CNN and for LLama3.2-Vision 11B](tests/output/public/calibration_curves.png)

> [!NOTE]  
> See details in the associated [wiki](https://github.com/IonMich/batch-doc-vqa/wiki/Row-of-Digits-OCR:-OpenCV-CNN-versus-LLMs) article.

This repository uses Large Language Models with vision capabilities to extract information from collections of documents. The goal is to create a fully local pipeline that runs on a single machine, and can be used to extract information from document collections for usage in downstream tasks.

## Benchmarks

Our small test dataset (`./imgs/quiz11-presidents.pdf`) consists of 32 documents representing Physics quizzes and the task is to match them to the test students who took the quiz via their 8-digit university ID and, optionally, their names (`./tests/data/test_ids.csv`). We have already saturated our test dataset with 100% statistically confident detections, but more optimizations are explored to decrease inference cost. You can find more details [here](https://github.com/IonMich/batch-doc-vqa/wiki/Row-of-Digits-OCR:-OpenCV-CNN-versus-LLMs). Currently the best performing pipeline is one that uses `outlines` to enforce JSON schemas on the model's responses, and the Qwen2-VL series of models. It uses less than 5GB of VRAM and completes in about 2 minutes on an RTX 3060 Ti. See the code [here](./outlines_quiz.py). The pipeline has been tested only on Ubuntu 22.04 with an RTX 3060 Ti and 8GB of VRAM.

|                         | OpenCV+CNN     | outlines + VLM                | outlines + VLM                | VLM w/ structured output | VLM w/ structured output |
|:------------------------|:---------------|:------------------------------|:------------------------------|:----------------------|:----------------------|
| LLM model               | N/A            | outlines + Qwen2-VL-2B-Instruct | outlines + SmolVLM          | gpt-4o-mini           | gemini-2.0-flash-exp  |
| LLM model size          | N/A            | 2B                            | 2B                            | ??                    | ??                    |
| Open-weights            | N/A            | Yes (Apache 2.0 License)      | Yes (Apache 2.0 License)      | No                    | No                    |
| # samples               | 1              | 1                             | 1                             | 1                     | 1                     |
| logits available        | Yes            | No                            | No                            | Yes (unused)          | Yes (unused)          |
| regex pattern           | N/A            | Yes                           | Yes                           | No                    | No                    |
| digit_top1              | 85.16%         | 98.44%                        | 68.35%                        | 26.95%                | 99.22%                |
| digit_top2              | 90.62%         | N/A                           | N/A                           | N/A                   | N/A                   |
| digit_top3              | 94.14%         | N/A                           | N/A                           | N/A                   | N/A                   |
| 8-digit id_top1         | N/A            | 90.63%                        | 53.13%                        | 50.00%                | 93.75%                |
| lastname_top1           | N/A            | 100%                          | 93.75%                        | 93.75%                | 93.75%                |
| Detect Type             | ID (1)         | LastName (2) + ID (1)         | LastName (2) + ID (1)         | LastName (2) + ID (1) | LastName (2) + ID (1) |
| ID Avg $d_\mathrm{Lev}$ | ?              | 0.1250                        | 2.3750                        | 0.6563                | 0.0625                |
| Lastname Avg $d_\mathrm{Lev}$ | N/A      | 0.0000                        | 0.0938                        | 0.156250              | 0.0625                |
| Docs detected           | 90.62% (29/32) | 100.00% (32/32)               | 68.75% (22/32)                | 100% (32/32)          | 100% (32/32)          |
| Runtime                 | ~ 1 second     | ~ 2 minutes (RTX 3060 Ti 8GB) | ~ 2 minutes (RTX 3060 Ti 8GB) | ~5 minutes (sequential) | ~5 minutes (sequential) |

## Outlines + Qwen2-VL

### Installation

1. Install `outlines` (Structured Generation) and `transformers_vision` (Vission LLM backbone) via the official installation instructions in the `outlines` Wiki [here](https://dottxt-ai.github.io/outlines/latest/installation/) and [here](https://dottxt-ai.github.io/outlines/latest/reference/models/transformers_vision/).

### Example Usage

The pipeline is split into multiple Python Scripts with the following command line usage:

```bash
    python pdf_to_imgs.py --filepath imgs/quiz11-presidents.pdf --pages_i 4 --dpi 300 --output_dir imgs/q11/
    python outlines_quiz.py
    python string_matching.py
```

Use `--help` to see the full list of options for each script.

## [OLD] Ollama + Llama3.2-Vision 11B

### Ollama Installation

1. First, install Ollama via the official [installation instructions](https://ollama.com/).

2. Pull the Llama3.2-Vision 11B model (or another model of your choice) on the Ollama server:

    ```bash
    ollama pull llama3.2-vision
    ```

3. Install the Python Ollama wrapper. You can do this via pip in a dedicated [conda](https://docs.anaconda.com/miniconda/) environment:

    ```bash
    conda create -n ollama python=3.11
    conda activate ollama
    pip install ollama
    ```

4. Clone this repository. There are no additional dependencies.

    ```bash
    git clone https://github.com/IonMich/batch-doc-vqa
    cd batch-doc-vqa
    ```

5. Start the Ollama server. On MacOS this can be done by simply opening the Ollama app. On Linux run `ollama server` in the terminal.

6. Check the `SYSTEM_MESSAGES` variable in `llamavision.py`. This are the prompts that the model will use to generate its responses (by default the prompt at index `0`). Leave the prompt unchanged if you want to check the installation on the test images. Feel free to add your own images to the `imgs` directory, or to change the `--filepath` command line argument to point to a different directory.

7. That's it! You're ready to use the pipeline:

    ```bash
    python llamavision.py --filepath imgs/sub-page-3.png --n_trials 10
    ```

### Usage

The Ollama pipeline is a Python scriptwith the following command line usage:

```bash
    python llamavision.py [-h] [--filepath FILEPATH] [--pattern PATTERN] [--n_trials N_TRIALS] [--system SYSTEM] [--model MODEL] [--no-stream] [--top_k TOP_K]
```

The pipeline has the following options:

- `--filepath`: (str) The path to the directory containing the images to be processed. If the path is a PNG file, the pipeline will process that single image. If the path is a directory, the pipeline will process (by default) all images in that directory.

- `--pattern`: (str) A string that is used to filter the images in the directory. E.g. `--pattern "page-3"` will only process PNG files that contain the string `page-3` in their filename. By default, this is set to `""`, which means that all images in the directory will be processed. This option is ignored if the `--filepath` is a single PNG file.

- `--n_trials`: (int) The number of trials to run for each image. This can be useful to do multi-shot inference on the same image, or to get a sense of the variance in the model's predictions.

- `--system`: (str) The system prompt index to use, which selects the prompt from a list of system prompts. By default, this is set to 0. See `SYSTEM_PROMPTS` in `llamavision.py` for the list of some example system prompts, and replace these with your own prompts.

- `--model`: (str) The model to use. By default, this is set to `llama3.2-vision`, but any LLM with vision capabilities can be used.

- `--top_k`: (int) The number of top answers to return. By default, this is set to 10. Lowering this number decreases creativity and variance in the model's predictions. Note that this is less than the Ollama default of 40.

- `--no-stream`: (flag) If set, the pipeline will not stream the LLM responses in chunks. Instead, the pipeline will wait for the model to finish processing the image before printing the responses. Default (when this flag is omitted) is to stream the responses.

## Motivations

Recent advances in LLM modelling have made it conceivable to build a quantifiably reliable pipeline to extract information in bulk from documents:

- Well formatted JSON can be fully enforced. In fact, using [context-free grammars](https://stackoverflow.com/a/6713333/10119867), precise JSON schemas can be enforced in language models that support structured responses (e.g. see [OpenAI's blog post](https://openai.com/index/introducing-structured-outputs-in-the-api/)).
- OpenAI's `GPT4 o1-preview` [appears to be well-calibrated](https://openai.com/index/introducing-simpleqa/), i.e. the frequency of its answers to fact-seeking questions is a good proxy for their accuracy. This creates the possibility to sample multiple times from the model to infer probabilities of each distinct answer. It is unclear however how well this calibration generalizes to any open-source models. It is also unclear if the purely textual SimpleQA task is a good proxy for text+vision task.
- The latest open-source models, such as the (Q4 quantized) Llama3.2-Vision 11B, show good performance on a variety of tasks, including document DocVQA, when compared to closed-source models like GPT-4. The [OCRBench Space](https://huggingface.co/spaces/echo840/ocrbench-leaderboard) on Huggingface has a nice summary of their performance on various OCR tasks.
- Hardware with acceptable memory bandwidth and large-enough memory capacity for LLM tasks is becoming more affordable.
