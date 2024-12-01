# conda activate outlines
from collections import defaultdict
import json
import os

import re


from PIL import Image
import torch

from transformers import AutoProcessor

# Qwen
from transformers import Qwen2VLForConditionalGeneration

# outlines
import outlines
from pydantic import BaseModel, Field
from typing import Optional

# Rich for pretty printing
from rich import print

UNIVERSITY_ID_LEN = 8
UNIVERSITY_ID_PATTERN = r"\d{" + str(UNIVERSITY_ID_LEN) + "}"
UNIVERSITY_ID_ALIAS = "ufid"
SECTION_NUMBER_LEN = 5
SECTION_NUMBER_PATTERN = r"\d{" + str(SECTION_NUMBER_LEN) + "}"


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
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

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


class QuizSubmissionSummary(BaseModel):
    # student_first_name: str
    # student_last_name: str
    student_full_name: str = Field(
        description="Full name of the student in the format First Last"
    )
    university_id: str = Field(
        # try also literal list of UFIDs
        pattern=UNIVERSITY_ID_PATTERN,
        alias=UNIVERSITY_ID_ALIAS,
        description=f"{UNIVERSITY_ID_LEN}-digit {UNIVERSITY_ID_ALIAS.capitalize()} of the student",
    )
    # section_number: str = Field(
    #     pattern=SECTION_NUMBER_PATTERN,
    #     description="Five digit section number where the student is enrolled",
    # )
    # problem_number: Optional[int]
    # problem_description: Optional[str] = Field(
    #     description="Description of the problem the student is tasked to solve"
    # )
    # student_work_latex: Optional[str] = Field(
    #     description="Student's handwritten work converted to LaTeX"
    # )
    # student_final_answer: Optional[float] = Field(
    #     description="Student's final handwritten answer. Sometimes the answer is boxed by the student."
    # )
    # date: Optional[str] = Field(
    #     pattern=r"\d{4}-\d{2}-\d{2}", description="Date in the format YYYY-MM-DD"
    # )


def outlines_qwen():
    # Load + resize the image
    pages = [1, 3]
    folder = "imgs/q11/"
    pattern = r"doc-\d+-page-[" + "".join([str(p) for p in pages]) + "]-[A-Z0-9]+.png"
    images = get_images(folder, pattern)

    model_uri = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
    model_class = Qwen2VLForConditionalGeneration
    model = outlines.models.transformers_vision(
        model_uri,
        model_class=model_class,
        model_kwargs={
            "device_map": "cuda:0",
            "torch_dtype": torch.float16,
            "attn_implementation": "flash_attention_2",
        },
        processor_kwargs={
            "device": "cuda",  # set to "cpu" if you don't have a GPU
        },
    )

    messages = [
        {
            "role": "user",
            "content": [
                {
                    # The image is provided as a PIL Image object
                    "type": "image",
                    "image": "",
                },
                {
                    "type": "text",
                    "text": f"""You are an expert at grading student quizzes in physics courses.
                    Please extract the information from the student's submission. Be as detailed as possible --
                    missing or misreporting information is a crime.

                    Return the information in the following JSON schema:
                    {QuizSubmissionSummary.model_json_schema(by_alias=True)}
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
    quiz_summary_generator = outlines.generate.json(
        model,
        QuizSubmissionSummary,
        # Greedy sampling is a good idea for numeric
        # data extraction -- no randomness.
        # sampler=outlines.samplers.greedy(),
        sampler=outlines.samplers.multinomial(temperature=0.5),
    )

    # Generate the quiz submission summary
    results = defaultdict(list)
    n_samples = 1
    for imagepath, image in images.items():
        for _ in range(n_samples):
            result = quiz_summary_generator(prompt, [image])
            print(result.model_dump(mode="json", by_alias=True))
            results[imagepath].append(result.model_dump(mode="json"))
        print("\n")

    # save the results
    json_save_results(results, filepath="qwen2-VL-2B-results.json")


def json_save_results(results, filepath):
    # save the results
    with open(filepath, "w") as f:
        json.dump(results, f)


def json_load_results(filepath):
    with open(filepath, "r") as f:
        results = json.load(f)
    return results


if __name__ == "__main__":
    outlines_qwen()
