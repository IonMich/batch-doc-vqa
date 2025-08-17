import json
from collections import defaultdict
import time
from PIL import Image

from pydantic import BaseModel, Field
import dotenv

from google import genai
from google.genai import types


dotenv.load_dotenv()


# Import shared utilities from core
from ..core import filepath_to_base64, get_imagepaths, natural_sort_key


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


# Use the same prompt as OpenRouter for consistency
prompt = """Extract the student information from this quiz submission. Return ONLY valid JSON in this format:
{
    "student_full_name": "Full name of the student",
    "ufid": "8-digit UFID number if present, empty string if missing",
    "section_number": "5-digit section number"
}

Example:
{
    "student_full_name": "John Doe",
    "ufid": "12345678",
    "section_number": "11900"
}

If UFID is not visible in the image, use an empty string for ufid."""

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
        # Apply JSON parsing to clean up markdown formatting for all models
        json_str = parse_json(json_str)
        print(f"Cleaned JSON string: '{json_str}'")
        json_obj = json.loads(json_str)
        # replace ufid with university_id
        json_obj["university_id"] = json_obj.pop("ufid")
        print(json_obj)
        results[imagepath].append(json_obj)
        print("\n")
        time.sleep(sleep_time)

    # save the results
    json_save_results(results, filepath=f"tests/output/{model_name}-results.json")


def run_gemini_inference(model_id: str, 
                         temperature: float = 0.0):
    """Run Gemini inference on the quiz images."""
    import dotenv
    dotenv.load_dotenv()

    client = genai.Client()
    
    config = types.GenerateContentConfig(
        # response_mime_type="application/json", 
        # response_schema=QuizSubmissionSummary,
        temperature=temperature,
        thinking_config=types.ThinkingConfig(
            include_thoughts=False,
            thinking_budget=-1, # dynamic thinking
        ),
    )
    
    # Use global imagepaths from module
    parse_images(client, model_id, config, imagepaths)


if __name__ == "__main__":
    run_gemini_inference(model_id="gemini-2.5-flash-lite", temperature=0.0)
