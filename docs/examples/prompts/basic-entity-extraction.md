You are extracting named entities from a single scanned document page.

Return only a JSON object that matches the provided JSON Schema exactly.

Extraction target:
- People
- Organizations
- Locations
- Dates
- ID-like numbers

Output rules:
- Use key `entities` as an array of extracted entities.
- For each entity, include:
  - `text`: exact text span as shown in the image
  - `label`: one of `PERSON`, `ORG`, `LOCATION`, `DATE`, `ID_NUMBER`, `OTHER`
  - `page`: positive integer page number
  - `confidence`: number between 0 and 1
- If no entities are visible, return `{"entities":[]}`.
- Do not include markdown, code fences, or explanation text.
