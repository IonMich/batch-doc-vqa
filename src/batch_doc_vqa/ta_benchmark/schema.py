"""Schema loading and validation helpers for TA benchmark labels."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from .constants import (
    DEFAULT_TA_SCHEMA_PATH,
    DEFAULT_TA_TAXONOMY_PATH,
    TA_SCHEMA_VERSION,
)

try:
    from jsonschema import Draft202012Validator
except Exception:  # pragma: no cover
    Draft202012Validator = None  # type: ignore[assignment]

_EVIDENCE_ID_RE = re.compile(r"^ev[A-Za-z0-9_-]+$")
_EVIDENCE_KINDS = {
    "problem_description",
    "problem_number",
    "figure",
    "work_region",
    "rubric_evidence",
    "feedback_anchor",
    "other",
}
_REVIEW_STATUSES = {"draft", "in_review", "verified"}
_PROPOSAL_ACTIONS = {"", "accepted", "edited", "rejected"}


@dataclass(frozen=True)
class ValidationIssue:
    """One schema or semantic validation issue."""

    level: str
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True)
class ValidationResult:
    """Validation result for one label document."""

    is_valid: bool
    issues: list[ValidationIssue]


def _path_to_str(path: Sequence[Any]) -> str:
    if not path:
        return "$"
    chunks: list[str] = ["$"]
    for part in path:
        if isinstance(part, int):
            chunks.append(f"[{part}]")
        else:
            chunks.append(f".{part}")
    return "".join(chunks)


def _load_json(path: str | Path) -> Any:
    with open(Path(path).expanduser().resolve(strict=False), "r", encoding="utf-8") as f:
        return json.load(f)


def _add_issue(issues: list[ValidationIssue], *, code: str, message: str, path: str) -> None:
    issues.append(ValidationIssue(level="error", code=code, message=message, path=path))


def _validate_non_empty_string(
    value: Any,
    *,
    issues: list[ValidationIssue],
    path: str,
    code: str,
    message: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        _add_issue(issues, code=code, message=message, path=path)


def _validate_evidence_id(value: Any, *, issues: list[ValidationIssue], path: str) -> None:
    if not isinstance(value, str) or not _EVIDENCE_ID_RE.match(value):
        _add_issue(
            issues,
            code="schema",
            message="evidence_id values must match ^ev[A-Za-z0-9_-]+$",
            path=path,
        )


def _validate_minimal_schema_payload(payload: dict[str, Any]) -> list[ValidationIssue]:
    """Fallback schema checks when jsonschema is unavailable."""
    issues: list[ValidationIssue] = []

    required_root = [
        "schema_version",
        "doc_id",
        "submission_id",
        "assignment_id",
        "template_version_id",
        "evidence_regions",
        "problems",
        "review",
    ]
    for key in required_root:
        if key not in payload:
            _add_issue(
                issues,
                code="schema",
                message=f"'{key}' is a required property",
                path="$",
            )

    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str):
        _add_issue(
            issues,
            code="schema",
            message="schema_version must be a string",
            path="$.schema_version",
        )

    doc_id = payload.get("doc_id")
    if not isinstance(doc_id, int) or doc_id < 0:
        _add_issue(
            issues,
            code="schema",
            message="doc_id must be an integer >= 0",
            path="$.doc_id",
        )

    _validate_non_empty_string(
        payload.get("submission_id"),
        issues=issues,
        path="$.submission_id",
        code="schema",
        message="submission_id must be a non-empty string",
    )

    assignment_id = payload.get("assignment_id")
    if not isinstance(assignment_id, int) or assignment_id < 1:
        _add_issue(
            issues,
            code="schema",
            message="assignment_id must be an integer >= 1",
            path="$.assignment_id",
        )

    if not isinstance(payload.get("template_version_id"), str):
        _add_issue(
            issues,
            code="schema",
            message="template_version_id must be a string",
            path="$.template_version_id",
        )

    evidence_regions = payload.get("evidence_regions", [])
    if not isinstance(evidence_regions, list):
        _add_issue(
            issues,
            code="schema",
            message="evidence_regions must be an array",
            path="$.evidence_regions",
        )
        evidence_regions = []
    for idx, row in enumerate(evidence_regions):
        base = f"$.evidence_regions[{idx}]"
        if not isinstance(row, dict):
            _add_issue(issues, code="schema", message="evidence region must be an object", path=base)
            continue
        for key in ("evidence_id", "page", "bbox", "kind"):
            if key not in row:
                _add_issue(
                    issues,
                    code="schema",
                    message=f"'{key}' is a required property",
                    path=base,
                )
        _validate_evidence_id(row.get("evidence_id"), issues=issues, path=f"{base}.evidence_id")

        page = row.get("page")
        if not isinstance(page, int) or page < 1:
            _add_issue(
                issues,
                code="schema",
                message="page must be an integer >= 1",
                path=f"{base}.page",
            )

        bbox = row.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            _add_issue(
                issues,
                code="schema",
                message="bbox must be an array with exactly 4 numbers",
                path=f"{base}.bbox",
            )
        else:
            for b_idx, value in enumerate(bbox):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    _add_issue(
                        issues,
                        code="schema",
                        message="bbox entries must be numbers in [0, 1]",
                        path=f"{base}.bbox[{b_idx}]",
                    )
                    continue
                if number < 0.0 or number > 1.0:
                    _add_issue(
                        issues,
                        code="schema",
                        message="bbox entries must be numbers in [0, 1]",
                        path=f"{base}.bbox[{b_idx}]",
                    )

        kind = row.get("kind")
        if not isinstance(kind, str) or kind not in _EVIDENCE_KINDS:
            _add_issue(
                issues,
                code="schema",
                message="kind must be one of the defined evidence kinds",
                path=f"{base}.kind",
            )
        if "label" in row and not isinstance(row.get("label"), str):
            _add_issue(
                issues,
                code="schema",
                message="label must be a string",
                path=f"{base}.label",
            )

    problems = payload.get("problems", [])
    if not isinstance(problems, list):
        _add_issue(issues, code="schema", message="problems must be an array", path="$.problems")
        problems = []
    for idx, row in enumerate(problems):
        base = f"$.problems[{idx}]"
        if not isinstance(row, dict):
            _add_issue(issues, code="schema", message="problem rows must be objects", path=base)
            continue
        for key in (
            "problem_uid",
            "problem_number",
            "description_text",
            "description_evidence_ids",
            "figure_evidence_ids",
        ):
            if key not in row:
                _add_issue(
                    issues,
                    code="schema",
                    message=f"'{key}' is a required property",
                    path=base,
                )
        _validate_non_empty_string(
            row.get("problem_uid"),
            issues=issues,
            path=f"{base}.problem_uid",
            code="schema",
            message="problem_uid must be a non-empty string",
        )
        if not isinstance(row.get("problem_number"), str):
            _add_issue(
                issues,
                code="schema",
                message="problem_number must be a string",
                path=f"{base}.problem_number",
            )
        if not isinstance(row.get("description_text"), str):
            _add_issue(
                issues,
                code="schema",
                message="description_text must be a string",
                path=f"{base}.description_text",
            )
        for key in ("description_evidence_ids", "figure_evidence_ids"):
            refs = row.get(key)
            if not isinstance(refs, list):
                _add_issue(
                    issues,
                    code="schema",
                    message=f"{key} must be an array",
                    path=f"{base}.{key}",
                )
                continue
            for r_idx, ref in enumerate(refs):
                _validate_evidence_id(ref, issues=issues, path=f"{base}.{key}[{r_idx}]")

    tier2 = payload.get("tier2")
    if tier2 is not None:
        if not isinstance(tier2, dict):
            _add_issue(issues, code="schema", message="tier2 must be an object", path="$.tier2")
        else:
            rubric_rows = tier2.get("rubric_scores", [])
            if rubric_rows is not None and not isinstance(rubric_rows, list):
                _add_issue(
                    issues,
                    code="schema",
                    message="rubric_scores must be an array",
                    path="$.tier2.rubric_scores",
                )
                rubric_rows = []
            for idx, row in enumerate(rubric_rows or []):
                base = f"$.tier2.rubric_scores[{idx}]"
                if not isinstance(row, dict):
                    _add_issue(issues, code="schema", message="rubric row must be an object", path=base)
                    continue
                for key in (
                    "problem_uid",
                    "criterion_id",
                    "max_points",
                    "awarded_points",
                    "error_tags",
                    "evidence_ids",
                ):
                    if key not in row:
                        _add_issue(
                            issues,
                            code="schema",
                            message=f"'{key}' is a required property",
                            path=base,
                        )

                _validate_non_empty_string(
                    row.get("problem_uid"),
                    issues=issues,
                    path=f"{base}.problem_uid",
                    code="schema",
                    message="problem_uid must be a non-empty string",
                )
                _validate_non_empty_string(
                    row.get("criterion_id"),
                    issues=issues,
                    path=f"{base}.criterion_id",
                    code="schema",
                    message="criterion_id must be a non-empty string",
                )

                for key in ("max_points", "awarded_points"):
                    try:
                        value = float(row.get(key))
                    except (TypeError, ValueError):
                        _add_issue(
                            issues,
                            code="schema",
                            message=f"{key} must be a number >= 0",
                            path=f"{base}.{key}",
                        )
                        continue
                    if value < 0:
                        _add_issue(
                            issues,
                            code="schema",
                            message=f"{key} must be a number >= 0",
                            path=f"{base}.{key}",
                        )

                tags = row.get("error_tags")
                if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
                    _add_issue(
                        issues,
                        code="schema",
                        message="error_tags must be an array of strings",
                        path=f"{base}.error_tags",
                    )

                refs = row.get("evidence_ids")
                if not isinstance(refs, list):
                    _add_issue(
                        issues,
                        code="schema",
                        message="evidence_ids must be an array",
                        path=f"{base}.evidence_ids",
                    )
                else:
                    for r_idx, ref in enumerate(refs):
                        _validate_evidence_id(ref, issues=issues, path=f"{base}.evidence_ids[{r_idx}]")

            feedback_rows = tier2.get("feedback", [])
            if feedback_rows is not None and not isinstance(feedback_rows, list):
                _add_issue(
                    issues,
                    code="schema",
                    message="feedback must be an array",
                    path="$.tier2.feedback",
                )
                feedback_rows = []
            for idx, row in enumerate(feedback_rows or []):
                base = f"$.tier2.feedback[{idx}]"
                if not isinstance(row, dict):
                    _add_issue(issues, code="schema", message="feedback row must be an object", path=base)
                    continue
                for key in ("problem_uid", "comment", "specificity", "actionability", "evidence_ids"):
                    if key not in row:
                        _add_issue(
                            issues,
                            code="schema",
                            message=f"'{key}' is a required property",
                            path=base,
                        )

                _validate_non_empty_string(
                    row.get("problem_uid"),
                    issues=issues,
                    path=f"{base}.problem_uid",
                    code="schema",
                    message="problem_uid must be a non-empty string",
                )
                if not isinstance(row.get("comment"), str):
                    _add_issue(issues, code="schema", message="comment must be a string", path=f"{base}.comment")

                for key in ("specificity", "actionability", "correctness"):
                    if key not in row:
                        continue
                    value = row.get(key)
                    if not isinstance(value, int) or value not in {0, 1}:
                        _add_issue(
                            issues,
                            code="schema",
                            message=f"{key} must be 0 or 1",
                            path=f"{base}.{key}",
                        )

                refs = row.get("evidence_ids")
                if not isinstance(refs, list):
                    _add_issue(
                        issues,
                        code="schema",
                        message="evidence_ids must be an array",
                        path=f"{base}.evidence_ids",
                    )
                else:
                    for r_idx, ref in enumerate(refs):
                        _validate_evidence_id(ref, issues=issues, path=f"{base}.evidence_ids[{r_idx}]")

    review = payload.get("review")
    if not isinstance(review, dict):
        _add_issue(issues, code="schema", message="review must be an object", path="$.review")
    else:
        for key in ("annotator_id", "status", "updated_at_utc"):
            if key not in review:
                _add_issue(
                    issues,
                    code="schema",
                    message=f"'{key}' is a required property",
                    path="$.review",
                )
        _validate_non_empty_string(
            review.get("annotator_id"),
            issues=issues,
            path="$.review.annotator_id",
            code="schema",
            message="annotator_id must be a non-empty string",
        )
        status = review.get("status")
        if not isinstance(status, str) or status not in _REVIEW_STATUSES:
            _add_issue(
                issues,
                code="schema",
                message=f"status must be one of {sorted(_REVIEW_STATUSES)}",
                path="$.review.status",
            )
        updated_at_utc = review.get("updated_at_utc")
        if not isinstance(updated_at_utc, str) or not updated_at_utc.strip():
            _add_issue(
                issues,
                code="schema",
                message="updated_at_utc must be a date-time string",
                path="$.review.updated_at_utc",
            )
        proposal_action = review.get("proposal_action")
        if proposal_action is not None:
            if not isinstance(proposal_action, str) or proposal_action not in _PROPOSAL_ACTIONS:
                _add_issue(
                    issues,
                    code="schema",
                    message=f"proposal_action must be one of {sorted(_PROPOSAL_ACTIONS)}",
                    path="$.review.proposal_action",
                )

    return issues


def load_ta_schema(path: Optional[str] = None) -> dict[str, Any]:
    payload = _load_json(path or DEFAULT_TA_SCHEMA_PATH)
    if not isinstance(payload, dict):
        raise ValueError("TA schema must be a JSON object.")
    return payload


def load_error_taxonomy(path: Optional[str] = None) -> set[str]:
    payload = _load_json(path or DEFAULT_TA_TAXONOMY_PATH)
    if not isinstance(payload, dict):
        raise ValueError("Error taxonomy must be a JSON object.")

    tags = payload.get("tags", [])
    if not isinstance(tags, list):
        return set()

    values: set[str] = set()
    for tag in tags:
        if not isinstance(tag, dict):
            continue
        tag_id = tag.get("id")
        if isinstance(tag_id, str) and tag_id.strip():
            values.add(tag_id.strip())
    return values


def collect_evidence_ids(label: dict[str, Any]) -> set[str]:
    ids: set[str] = set()
    regions = label.get("evidence_regions", [])
    if not isinstance(regions, list):
        return ids
    for region in regions:
        if not isinstance(region, dict):
            continue
        evidence_id = region.get("evidence_id")
        if isinstance(evidence_id, str) and evidence_id.strip():
            ids.add(evidence_id.strip())
    return ids


def _validate_schema_payload(
    payload: dict[str, Any],
    schema: dict[str, Any],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    if Draft202012Validator is None:  # pragma: no cover
        issues.extend(_validate_minimal_schema_payload(payload))
        return issues

    validator = Draft202012Validator(schema)
    for err in sorted(validator.iter_errors(payload), key=lambda item: list(item.path)):
        issues.append(
            ValidationIssue(
                level="error",
                code="schema",
                message=err.message,
                path=_path_to_str(list(err.path)),
            )
        )
    return issues


def _validate_bbox_geometry(payload: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    regions = payload.get("evidence_regions", [])
    if not isinstance(regions, list):
        return issues

    for idx, region in enumerate(regions):
        if not isinstance(region, dict):
            continue
        bbox = region.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(value) for value in bbox]
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1:
            issues.append(
                ValidationIssue(
                    level="error",
                    code="bbox_degenerate",
                    message=f"bbox must satisfy x2>x1 and y2>y1 for evidence_regions[{idx}]",
                    path=f"$.evidence_regions[{idx}].bbox",
                )
            )
    return issues


def _validate_reference_integrity(
    payload: dict[str, Any],
    *,
    known_error_tags: Optional[set[str]],
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    evidence_ids = collect_evidence_ids(payload)

    problems = payload.get("problems", [])
    problem_ids: set[str] = set()
    if isinstance(problems, list):
        for p_idx, problem in enumerate(problems):
            if not isinstance(problem, dict):
                continue
            problem_uid = problem.get("problem_uid")
            if isinstance(problem_uid, str) and problem_uid.strip():
                problem_ids.add(problem_uid.strip())

            for key in ("description_evidence_ids", "figure_evidence_ids"):
                refs = problem.get(key, [])
                if not isinstance(refs, list):
                    continue
                for ref in refs:
                    if not isinstance(ref, str) or not ref.strip():
                        continue
                    if ref.strip() not in evidence_ids:
                        issues.append(
                            ValidationIssue(
                                level="error",
                                code="unknown_evidence_ref",
                                message=f"{key} references unknown evidence_id '{ref}'",
                                path=f"$.problems[{p_idx}].{key}",
                            )
                        )

    tier2 = payload.get("tier2", {})
    if not isinstance(tier2, dict):
        return issues

    rubric_scores = tier2.get("rubric_scores", [])
    if isinstance(rubric_scores, list):
        for r_idx, row in enumerate(rubric_scores):
            if not isinstance(row, dict):
                continue
            awarded = row.get("awarded_points")
            max_points = row.get("max_points")
            try:
                awarded_f = float(awarded)
                max_f = float(max_points)
            except (TypeError, ValueError):
                continue
            if awarded_f < max_f:
                refs = row.get("evidence_ids", [])
                if not isinstance(refs, list) or not [r for r in refs if isinstance(r, str) and r.strip()]:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="missing_deduction_evidence",
                            message=(
                                "Each deduction (awarded_points < max_points) must cite at least one evidence reference."
                            ),
                            path=f"$.tier2.rubric_scores[{r_idx}].evidence_ids",
                        )
                    )

            refs = row.get("evidence_ids", [])
            if isinstance(refs, list):
                for ref in refs:
                    if not isinstance(ref, str) or not ref.strip():
                        continue
                    if ref.strip() not in evidence_ids:
                        issues.append(
                            ValidationIssue(
                                level="error",
                                code="unknown_evidence_ref",
                                message=f"rubric_scores references unknown evidence_id '{ref}'",
                                path=f"$.tier2.rubric_scores[{r_idx}].evidence_ids",
                            )
                        )

            problem_uid = row.get("problem_uid")
            if isinstance(problem_uid, str) and problem_uid.strip():
                if problem_uid.strip() not in problem_ids:
                    issues.append(
                        ValidationIssue(
                            level="error",
                            code="unknown_problem_ref",
                            message=f"rubric_scores references unknown problem_uid '{problem_uid}'",
                            path=f"$.tier2.rubric_scores[{r_idx}].problem_uid",
                        )
                    )

            tags = row.get("error_tags", [])
            if isinstance(tags, list) and known_error_tags:
                for tag in tags:
                    if not isinstance(tag, str) or not tag.strip():
                        continue
                    if tag.strip() not in known_error_tags:
                        issues.append(
                            ValidationIssue(
                                level="error",
                                code="unknown_error_tag",
                                message=f"Unknown error tag '{tag}'",
                                path=f"$.tier2.rubric_scores[{r_idx}].error_tags",
                            )
                        )

    feedback_rows = tier2.get("feedback", [])
    if isinstance(feedback_rows, list):
        for f_idx, row in enumerate(feedback_rows):
            if not isinstance(row, dict):
                continue
            problem_uid = row.get("problem_uid")
            if not isinstance(problem_uid, str) or not problem_uid.strip():
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="missing_feedback_problem_ref",
                        message="Each feedback row must link to a problem_uid.",
                        path=f"$.tier2.feedback[{f_idx}].problem_uid",
                    )
                )
            elif problem_uid.strip() not in problem_ids:
                issues.append(
                    ValidationIssue(
                        level="error",
                        code="unknown_problem_ref",
                        message=f"feedback references unknown problem_uid '{problem_uid}'",
                        path=f"$.tier2.feedback[{f_idx}].problem_uid",
                    )
                )

            refs = row.get("evidence_ids", [])
            if isinstance(refs, list):
                for ref in refs:
                    if not isinstance(ref, str) or not ref.strip():
                        continue
                    if ref.strip() not in evidence_ids:
                        issues.append(
                            ValidationIssue(
                                level="error",
                                code="unknown_evidence_ref",
                                message=f"feedback references unknown evidence_id '{ref}'",
                                path=f"$.tier2.feedback[{f_idx}].evidence_ids",
                            )
                        )

    return issues


def validate_label_payload(
    payload: dict[str, Any],
    *,
    schema: Optional[dict[str, Any]] = None,
    known_error_tags: Optional[set[str]] = None,
) -> ValidationResult:
    """Validate one TA label payload with schema + semantic checks."""
    active_schema = schema or load_ta_schema()
    issues: list[ValidationIssue] = []

    issues.extend(_validate_schema_payload(payload, active_schema))
    issues.extend(_validate_bbox_geometry(payload))
    issues.extend(
        _validate_reference_integrity(
            payload,
            known_error_tags=known_error_tags,
        )
    )

    schema_version = payload.get("schema_version")
    if schema_version != TA_SCHEMA_VERSION:
        issues.append(
            ValidationIssue(
                level="error",
                code="schema_version_mismatch",
                message=f"schema_version must equal '{TA_SCHEMA_VERSION}'",
                path="$.schema_version",
            )
        )

    is_valid = not any(issue.level == "error" for issue in issues)
    return ValidationResult(is_valid=is_valid, issues=issues)


def validate_label_file(
    path: str | Path,
    *,
    schema: Optional[dict[str, Any]] = None,
    known_error_tags: Optional[set[str]] = None,
) -> ValidationResult:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        return ValidationResult(
            is_valid=False,
            issues=[
                ValidationIssue(
                    level="error",
                    code="invalid_top_level",
                    message="Label file must contain a top-level object.",
                    path="$",
                )
            ],
        )
    return validate_label_payload(payload, schema=schema, known_error_tags=known_error_tags)


def iter_label_files(labels_dir: str | Path) -> Iterable[Path]:
    root = Path(labels_dir).expanduser().resolve(strict=False)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*.json") if path.is_file())
