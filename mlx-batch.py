import glob
import os

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config

# Load the model
# model_path = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
model_path = "mlx-community/Qwen2-VL-2B-Instruct-bf16"
model, processor = load(model_path)
config = load_config(model_path)

# Prepare input
# image = ["http://images.cocodataset.org/val2017/000000039769.jpg"]

image_paths = glob.glob(os.path.join("imgs/q11/", "*.png"))
# # only first image
# image_path = image_paths[0]
# print(image_path)
# image = [image_paths[0]]

from pydantic import BaseModel, Field

UNIVERSITY_ID_LEN = 8
UNIVERSITY_ID_PATTERN = f"^[0-9]{{{UNIVERSITY_ID_LEN}}}$"
UNIVERSITY_ID_ALIAS = "ufid"
SECTION_NUMBER_PATTERN = r"^\d{5}$"

class QuizSubmissionSummary(BaseModel):
    # student_first_name: str
    # student_last_name: str
    student_full_name: str = Field(
        description="Full name of the student in the format First Last"
    )
    university_id: str = Field(
        # try also literal list of UFIDs
        # pattern=UNIVERSITY_ID_PATTERN,
        alias=UNIVERSITY_ID_ALIAS,
        description=f"{UNIVERSITY_ID_LEN}-digit {UNIVERSITY_ID_ALIAS.capitalize()} of the student. If missing, report an empty string",
    )
    section_number: str = Field(
        # pattern=SECTION_NUMBER_PATTERN,
        description="5-digit section number of the student. If missing, report an empty string",
    )


prompt = f"""You are an expert at grading student quizzes in physics courses.
    Please extract the information from the student's submission. Reply only with the requested JSON schema, nothing else. Do not overthink the problem.

    Return the information in the following JSON schema:
    {QuizSubmissionSummary.model_json_schema(by_alias=True)}
"""
# prompt = "What is the handwritten full name of the person in the image?"

formatted_prompt = apply_chat_template(
    processor, config, prompt, num_images=1
)
# Apply chat template
for i in range(len(image_paths)):
    image_path = image_paths[i]
    # keep only those with "page-3" in the name
    if "page-3" not in image_path:
        continue
    image = [image_path]

    print(image_path)

    # Generate output
    output = generate(model, processor, formatted_prompt, image, verbose=True)
    print(output)