import os
import base64
from collections import defaultdict
import re
import json
import dotenv

from openai import OpenAI

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


def json_save_results(results, filepath):
    # save the results
    with open(filepath, "w") as f:
        json.dump(results, f)


pages = [1, 3]
folder = "imgs/q11/"
pattern = r"doc-\d+-page-[" + "".join([str(p) for p in pages]) + "]-[A-Z0-9]+.png"
imagepaths = get_imagepaths(folder, pattern)


def create_completion(client, model_name, config, imagepath):
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "text": "You are an expert at grading student quizzes in physics courses.\nPlease extract the information from the student's submission. Be as detailed as possible.\n\nReturn the information in the following JSON schema:\n{\n    'properties': {\n        'student_full_name': {'description': 'Full name of the student in the format First Last', 'title': 'Student Full Name', 'type': 'string'},\n        'ufid': {'description': '8-digit Ufid of the student', 'pattern': '\\\\d{8}', 'title': 'Ufid', 'type': 'string'},\n        'section_number': {'description': '5-digit section number of the student', 'pattern': '\\\\d{5}', 'title': 'Section Number', 'type': 'string'}\n    },\n    'required': ['student_full_name', 'ufid', 'section_number'],\n    'title': 'QuizSubmissionSummary',\n    'type': 'object'\n}",
                        "type": "text",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{filepath_to_base64(imagepath)}",
                        },
                    },
                ],
            }
        ],
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "QuizSubmissionSummary",
                "schema": {
                    "type": "object",
                    "required": ["student_full_name", "ufid", "section_number"],
                    "properties": {
                        "ufid": {
                            "type": "string",
                            "description": "8-digit UFID of the student",
                        },
                        "section_number": {
                            "type": "string",
                            "description": "5-digit section number of the student",
                        },
                        "student_full_name": {
                            "type": "string",
                            "description": "Full name of the student in the format First Last",
                        },
                    },
                    "additionalProperties": False,
                },
                "strict": True,
            },
        },
        **config,
    )

    return response


def parse_images(client, model_name, config, imagepaths):
    results = defaultdict(list)
    imagepaths = imagepaths[:1]
    for imagepath in imagepaths:
        response = create_completion(client, model_name, config, imagepath)
        print(response.model_dump(mode="json"))
        json_str = (
            response.model_dump(mode="json")
            .get("choices")[0]
            .get("message")
            .get("content")
        )
        json_obj = json.loads(json_str)
        # replace ufid with university_id
        json_obj["university_id"] = json_obj.pop("ufid")
        results[imagepath].append(json_obj)
        print("\n")

    # save the results
    json_save_results(results, filepath=f"tests/output/{model_name}-results.json")


if __name__ == "__main__":
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    model_name = "gpt-4o-mini"
    config = {
        "temperature": 0,
        "top_p": 1,
        "max_tokens": 256,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }
    # client = OpenAI(
    #     api_key=os.environ["GENAI_API_KEY"],
    #     base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    # )
    # model_name = "gemini-2.0-flash-thinking-exp-1219"
    # config = {
    #     "temperature": 0,
    #     "top_p": 1,
    #     "max_tokens": 8192,
    # }
    parse_images(client, model_name, config, imagepaths)
