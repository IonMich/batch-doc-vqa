# Vision LLM - Batch Document VQA with structured responses

This repository uses Large Language Models with vision capabilities to extract information from collections of documents. The goal is to create a fully local pipeline that runs on a single machine, and can be used to extract information from document collections for usage in downstream tasks.

This repository is a work in progress. The best performing pipeline currently is one that uses Llama3.2-Vision as quantized in the Ollama project and is tested on MacOS with an M2 chip and 16GB of RAM.

## Installation

1. First, install Ollama via the official [installation instructions](https://ollama.com/).

2. Pull the Llama3.2-Vision 11B model (or another model of your choice) on the Ollama server:

    ```bash
    ollama pull llama3.2-vision
    ```

3. Start the Ollama server. On MacOS this can be done by simply opening the Ollama app. On Linux run `ollama server` in the terminal.

4. Check the `SYSTEM_MESSAGES` variable in `llamavision.py`. This are the prompts that the model will use to generate its responses (by default the prompt at index `0`). Leave the prompt unchanged if you want to check the installation on the test images.

5. That's it! You're ready to use the pipeline:

    ```bash
    python llamavision.py --filepath imgs/ --n_trials 10
    ```

## Usage

The pipeline has the following options:

- `--filepath`: (str) The path to the directory containing the images to be processed. If the path is a PNG file, the pipeline will process that single image. If the path is a directory, the pipeline will process (by default) all images in that directory.

- `--pattern`: (str) A string that is used to filter the images in the directory. E.g. `--pattern "page-3"` will only process PNG files that contain the string `page-3` in their filename. By default, this is set to `""`, which means that all images in the directory will be processed. This option is ignored if the `--filepath` is a single PNG file.

- `--n_trials`: (int) The number of trials to run for each image. This can be useful to do multi-shot inference on the same image, or to get a sense of the variance in the model's predictions.

- `--system`: (str) The system prompt index to use, which selects the prompt from a list of system prompts. By default, this is set to 0. See `SYSTEM_PROMPTS` in `llamavision.py` for the list of some example system prompts, and replace these with your own prompts.

- `--model`: (str) The model to use. By default, this is set to `llama3.2-vision`, but any LLM with vision capabilities can be used.

- `--top_k`: (int) The number of top answers to return. By default, this is set to 10. Lowering this number decreases creativity and variance in the model's predictions. Note that this is less than the Ollama default of 40.

- `--no-stream`: (flag) If set, the pipeline will not stream the LLM responses in chunks. Instead, the pipeline will wait for the model to finish processing the image before printing the responses. Default (when this flag is omitted) is to stream the responses.
