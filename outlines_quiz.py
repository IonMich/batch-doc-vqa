# conda activate batch
import argparse
from pydantic import BaseModel, Field

from batch_extract import get_images, outlines_vlm
# outlines

UNIVERSITY_ID_LEN = 8
UNIVERSITY_ID_PATTERN = r"\d{" + str(UNIVERSITY_ID_LEN) + "}"
UNIVERSITY_ID_ALIAS = "ufid"
SECTION_NUMBER_LEN = 5
SECTION_NUMBER_PATTERN = r"\d{" + str(SECTION_NUMBER_LEN) + "}"


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
    section_number: str = Field(
        pattern=SECTION_NUMBER_PATTERN,
        description="5-digit section number of the student",
    )
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Outlines Quiz Submission Parser")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2-VL-2B-Instruct",
        dest="model_uri",
        type=str,
        help="Hugging Face model URI for the vision-to-sequence model",
    )
    args = parser.parse_args()
    # model_uri = "HuggingFaceTB/SmolVLM-Instruct"
    # model_uri = "Qwen/Qwen2-VL-2B-Instruct-AWQ"
    folder = "imgs/q11/"
    pages = [1, 3]
    pattern = r"doc-\d+-page-[" + "".join([str(p) for p in pages]) + "]-[A-Z0-9]+.png"
    quiz_images = get_images(folder, pattern)
    user_message = "You are an expert at grading student quizzes in physics courses. Please extract the information from the student's submission. Be as detailed as possible."

    outlines_vlm(
        images=quiz_images,
        model_uri=args.model_uri,
        pydantic_model=QuizSubmissionSummary,
        user_message=user_message,
    )
