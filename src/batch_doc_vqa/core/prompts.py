"""
Centralized prompt templates for batch document VQA tasks.
This module contains all prompt templates used across different inference backends.
"""

# Main prompt template for student information extraction
STUDENT_EXTRACTION_PROMPT = """Extract the student information from this quiz submission. Return ONLY valid JSON in this format:
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