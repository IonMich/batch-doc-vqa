import os
import re
from collections import defaultdict
import json

from PIL import Image
from pydantic import BaseModel
from rich import print

import torch
from transformers import AutoProcessor, AutoModelForVision2Seq
import outlines


def get_imagepaths(folder, pattern):
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if re.match(pattern, file):
                images.append(os.path.join(root, file))
    # sort by integers in the filename
    images.sort(key=natural_sort_key)
    print(images)
    return images


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def get_images(folder, pattern):
    filepaths = get_imagepaths(folder, pattern)
    return {filepath: load_and_resize_image(filepath) for filepath in filepaths}


def load_and_resize_image(image_path, max_size=1024):
    """
    Load and resize an image while maintaining aspect ratio

    Args:
        image_path: Path to the image file
        max_size: Maximum dimension (width or height) of the output image

    Returns:
        PIL Image: Resized image
    """
    image = Image.open(image_path)

    # Get current dimensions
    width, height = image.size

    # Calculate scaling factor
    scale = min(max_size / width, max_size / height)

    # Only resize if image is larger than max_size
    if scale < 1:
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    return image


def json_save_results(results, filepath):
    # save the results
    with open(filepath, "w") as f:
        json.dump(results, f)


def json_load_results(filepath):
    with open(filepath, "r") as f:
        results = json.load(f)
    return results


def outlines_vlm(
    images,
    model_uri,
    pydantic_model: BaseModel,
    model_class=AutoModelForVision2Seq,
    user_message: str = "You are a helpful assistant",
):
    output_path = f"tests/output/{model_uri.replace('/', '-')}-results.json"

    has_cuda = torch.cuda.is_available()
    model = outlines.models.transformers_vision(
        model_uri,
        model_class=model_class,
        model_kwargs={
            "device_map": "auto" if has_cuda else "cpu",
            "torch_dtype": torch.float16 if has_cuda else torch.float32,
            "attn_implementation": "flash_attention_2" if has_cuda else "eager",
        },
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "",
                },
                {
                    "type": "text",
                    # NOTE: probably here by_alias should be False for best results,
                    # since we pass the model directly to `outlines.generate.json`
                    # so during generation the LLM sees the original model attributes.
                    # Currently `outlines.generate.json` uses `model_json_schema()`,
                    # which uses the default `by_alias=False`.
                    "text": f"""{user_message}

                    Return the information in the following JSON schema:
                    {pydantic_model.model_json_schema(by_alias=False)}
                """,
                },
            ],
        }
    ]

    # Convert the messages to the final prompt
    processor = AutoProcessor.from_pretrained(model_uri)
    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    extract_generator = outlines.generate.json(
        model,
        pydantic_model,
        # Greedy sampling is a good idea for numeric
        # data extraction -- no randomness.
        sampler=outlines.samplers.greedy(),
        # sampler=outlines.samplers.multinomial(temperature=0.5),
    )

    # Generate the quiz submission summary
    results = defaultdict(list)
    n_samples = 1
    for imagepath, image in images.items():
        for _ in range(n_samples):
            result = extract_generator(prompt, [image])
            print(result.model_dump(mode="json", by_alias=True))
            results[imagepath].append(result.model_dump(mode="json"))
        print("\n")

    # save the results
    json_save_results(results, filepath=output_path)

    return results
