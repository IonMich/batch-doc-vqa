You are extracting page-level TA benchmark signals from a single submission page image.

Return only one JSON object that matches the provided JSON Schema exactly.

This is a page-level extraction step (Protocol M Step A):
- identify problem numbers/descriptions
- identify associated figure regions
- output evidence regions and problem objects
- optionally extract student name / ID / section if present on this page

Output rules:
- If no relevant signals are visible, return empty arrays and empty strings.
- Use normalized bbox coordinates in [0,1] as `[x1, y1, x2, y2]`.
- Prefer stable `problem_uid` values using problem number when visible (e.g., `p_1`).
- Do not include markdown, code fences, or explanation text.
