import os
import re
import json
import base64
from collections import defaultdict
import time
from PIL import Image

from pydantic import BaseModel, Field
import dotenv

from google import genai
from google.genai import types


dotenv.load_dotenv()


def filepath_to_base64(filepath):
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_imagepaths(folder, pattern):
    images = []
    for root, _, files in os.walk(folder):
        for file in files:
            if re.match(pattern, file):
                images.append(os.path.join(root, file))
    # sort by integers in the filename
    images.sort(key=natural_sort_key)
    return images


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)
    ]


def parse_json(json_output):
    # Parsing out the markdown fencing
    lines = json_output.splitlines()
    for i, line in enumerate(lines):
        if line == "```json":
            json_output = "\n".join(
                lines[i + 1 :]
            )  # Remove everything before "```json"
            json_output = json_output.split("```")[
                0
            ]  # Remove everything after the closing "```"
            break  # Exit the loop once "```json" is found
    return json_output


def json_save_results(results, filepath):
    # save the results
    with open(filepath, "w") as f:
        json.dump(results, f)


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
    Please extract the information from the student's submission. Be as detailed as possible. Do not overthink the problem.

    Return the information in the following JSON schema:
    {QuizSubmissionSummary.model_json_schema(by_alias=True)}

    Example:
    ```json
    {{
        "student_full_name": "John Doe",
        "ufid": "12345678",
        "section_number": "12345"
    }}
    ```
"""

print(prompt)

pages = [1, 3]
folder = "imgs/q11/"
pattern = r"doc-\d+-page-[" + "".join([str(p) for p in pages]) + "]-[A-Z0-9]+.png"
imagepaths = get_imagepaths(folder, pattern)


def create_completion(client, model_name, config, imagepath):
    im = Image.open(imagepath)
    response = client.models.generate_content(
        model=model_name,
        contents=[
            im,
            prompt,
        ],
        config=config,
    )
    for part in response.candidates[0].content.parts:
        if part.thought:
            print(f"Model Thought:\n{part.text}\n")
        else:
            print(f"\nModel Response:\n{part.text}\n")
    return response


def parse_images(client, model_name, config, imagepaths):
    max_requests_per_minute = 10
    sleep_time = 60 / max_requests_per_minute + 1 # add 1 second to be safe
    results = defaultdict(list)
    imagepaths = imagepaths
    for imagepath in imagepaths:
        response = create_completion(client, model_name, config, imagepath)
        response_parts = (
            response.model_dump(mode="json")
            .get("candidates")[0]
            .get("content")
            .get("parts")
        )
        model_responses = [r for r in response_parts if not r.get("thought")]
        json_str = model_responses[0].get("text")
        if "thinking" in model_name:
            json_str = parse_json(json_str)
        # print(json_str)
        json_obj = json.loads(json_str)
        # replace ufid with university_id
        json_obj["university_id"] = json_obj.pop("ufid")
        print(json_obj)
        results[imagepath].append(json_obj)
        print("\n")
        time.sleep(sleep_time)

    # save the results
    json_save_results(results, filepath=f"tests/output/{model_name}-results.json")


if __name__ == "__main__":
    api_key = os.getenv("GEMINI_API_KEY")

    client = genai.Client(api_key=api_key, http_options={"api_version": "v1alpha"})
    model_id = "gemini-2.0-flash-thinking-exp-01-21"
    # model_id = "gemini-2.0-flash-exp"
    config = types.GenerateContentConfig(
        # response_mime_type="application/json", 
        # response_schema=QuizSubmissionSummary,
        temperature=0,
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
        ),
    )
    parse_images(client, model_id, config, imagepaths)
