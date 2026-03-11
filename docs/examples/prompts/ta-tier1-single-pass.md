You are extracting TA benchmark Tier 1 signals from a single submission page image.

Return only one JSON object that matches the provided JSON Schema exactly.

Goals:
- detect problem-related regions
- transcribe problem descriptions
- link figures to problems when possible
- optionally extract identity fields if visible on this page
- optionally provide a template guess (low confidence is acceptable)

Output rules:
- `student_full_name`, `university_id`, `section_number` can be empty strings when not visible.
- Use normalized bbox coordinates in [0,1] as `[x1, y1, x2, y2]`.
- `problem_uid` should be stable when possible (e.g., `p_1`, `p_2`).
- If unknown, leave arrays empty rather than guessing.
- Do not include markdown, code fences, or explanation text.
