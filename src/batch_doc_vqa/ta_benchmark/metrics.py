"""Metrics for TA benchmark scoring."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Optional


def levenshtein_distance(left: str, right: str) -> int:
    """Compute Levenshtein edit distance."""
    if left == right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    if not left:
        return len(right)

    prev = list(range(len(left) + 1))
    for r_idx, r_ch in enumerate(right, start=1):
        cur = [r_idx]
        for l_idx, l_ch in enumerate(left, start=1):
            if l_ch == r_ch:
                cur.append(prev[l_idx - 1])
            else:
                cur.append(1 + min(prev[l_idx - 1], prev[l_idx], cur[-1]))
        prev = cur
    return prev[-1]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _index_by_doc(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    docs: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        doc_id = _safe_int(row.get("doc_id"), default=-1)
        if doc_id < 0:
            continue
        docs[doc_id] = row
    return docs


def _extract_regions(doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows = doc.get("evidence_regions", [])
    if not isinstance(rows, list):
        return []
    result: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        bbox = row.get("bbox", [])
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox]
        except (TypeError, ValueError):
            continue
        result.append(
            {
                "page": _safe_int(row.get("page"), 0),
                "kind": str(row.get("kind", "") or "").strip(),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return result


def _iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _problem_map(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    problems = doc.get("problems", [])
    result: dict[str, dict[str, Any]] = {}
    if not isinstance(problems, list):
        return result
    for row in problems:
        if not isinstance(row, dict):
            continue
        uid = str(row.get("problem_uid", "") or "").strip()
        if not uid:
            continue
        result[uid] = row
    return result


def _figure_pairs(problem_map: dict[str, dict[str, Any]]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for uid, problem in problem_map.items():
        refs = problem.get("figure_evidence_ids", [])
        if not isinstance(refs, list):
            continue
        for ref in refs:
            if isinstance(ref, str) and ref.strip():
                pairs.add((uid, ref.strip()))
    return pairs


def _extract_error_tag_tuples(doc: dict[str, Any]) -> set[tuple[str, str, str]]:
    tuples: set[tuple[str, str, str]] = set()
    tier2 = doc.get("tier2", {})
    if not isinstance(tier2, dict):
        return tuples
    scores = tier2.get("rubric_scores", [])
    if not isinstance(scores, list):
        return tuples
    for row in scores:
        if not isinstance(row, dict):
            continue
        problem_uid = str(row.get("problem_uid", "") or "").strip()
        criterion = str(row.get("criterion_id", "") or "").strip()
        tags = row.get("error_tags", [])
        if not problem_uid or not criterion or not isinstance(tags, list):
            continue
        for tag in tags:
            if isinstance(tag, str) and tag.strip():
                tuples.add((problem_uid, criterion, tag.strip()))
    return tuples


def _extract_rubric_points(doc: dict[str, Any]) -> dict[tuple[str, str], float]:
    points: dict[tuple[str, str], float] = {}
    tier2 = doc.get("tier2", {})
    if not isinstance(tier2, dict):
        return points
    scores = tier2.get("rubric_scores", [])
    if not isinstance(scores, list):
        return points
    for row in scores:
        if not isinstance(row, dict):
            continue
        problem_uid = str(row.get("problem_uid", "") or "").strip()
        criterion = str(row.get("criterion_id", "") or "").strip()
        if not problem_uid or not criterion:
            continue
        points[(problem_uid, criterion)] = _safe_float(row.get("awarded_points"), 0.0)
    return points


def _extract_feedback_dimensions(doc: dict[str, Any]) -> dict[str, dict[str, int]]:
    dims_by_problem: dict[str, dict[str, int]] = {}
    tier2 = doc.get("tier2", {})
    if not isinstance(tier2, dict):
        return dims_by_problem
    rows = tier2.get("feedback", [])
    if not isinstance(rows, list):
        return dims_by_problem
    for row in rows:
        if not isinstance(row, dict):
            continue
        problem_uid = str(row.get("problem_uid", "") or "").strip()
        if not problem_uid:
            continue
        dims_by_problem[problem_uid] = {
            "specificity": _safe_int(row.get("specificity"), 0),
            "actionability": _safe_int(row.get("actionability"), 0),
            "correctness": _safe_int(row.get("correctness"), 1),
        }
    return dims_by_problem


def quadratic_weighted_kappa(y_true: list[float], y_pred: list[float]) -> float:
    if not y_true or not y_pred or len(y_true) != len(y_pred):
        return 0.0

    rounded_true = [round(value, 2) for value in y_true]
    rounded_pred = [round(value, 2) for value in y_pred]
    labels = sorted(set(rounded_true) | set(rounded_pred))
    if len(labels) <= 1:
        return 1.0
    idx = {label: i for i, label in enumerate(labels)}
    n = len(labels)
    total = float(len(rounded_true))

    obs = [[0.0 for _ in range(n)] for _ in range(n)]
    true_hist = [0.0 for _ in range(n)]
    pred_hist = [0.0 for _ in range(n)]
    for t, p in zip(rounded_true, rounded_pred):
        ti = idx[t]
        pi = idx[p]
        obs[ti][pi] += 1.0
        true_hist[ti] += 1.0
        pred_hist[pi] += 1.0

    expected = [[(true_hist[i] * pred_hist[j]) / total for j in range(n)] for i in range(n)]
    denom = float((n - 1) ** 2)
    if denom <= 0.0:
        return 1.0

    weighted_obs = 0.0
    weighted_exp = 0.0
    for i in range(n):
        for j in range(n):
            weight = ((i - j) ** 2) / denom
            weighted_obs += weight * (obs[i][j] / total)
            weighted_exp += weight * (expected[i][j] / total)
    if weighted_exp <= 0.0:
        return 1.0
    return 1.0 - (weighted_obs / weighted_exp)


def score_documents(
    labels: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute full TA benchmark metric bundle."""
    label_by_doc = _index_by_doc(labels)
    pred_by_doc = _index_by_doc(predictions)
    docs = sorted(label_by_doc.keys())

    region_tp = 0
    region_fp = 0
    region_fn = 0

    text_distance_sum = 0
    text_char_total = 0
    text_norm_sum = 0.0
    text_pair_count = 0

    figure_tp = 0
    figure_fp = 0
    figure_fn = 0

    template_correct = 0
    template_total = 0

    tag_tp = 0
    tag_fp = 0
    tag_fn = 0
    per_tag_counts: dict[str, Counter[str]] = defaultdict(Counter)

    rubric_abs_errors: list[float] = []
    rubric_exact = 0
    rubric_total = 0
    qwk_true: list[float] = []
    qwk_pred: list[float] = []

    feedback_dim_matches = Counter()
    feedback_dim_total = Counter()

    per_doc_rows: list[dict[str, Any]] = []

    for doc_id in docs:
        gt_doc = label_by_doc.get(doc_id, {})
        pred_doc = pred_by_doc.get(doc_id, {})

        # Region detection metrics
        gt_regions = _extract_regions(gt_doc)
        pred_regions = _extract_regions(pred_doc)
        matched_pred: set[int] = set()
        doc_tp = 0
        for gt_region in gt_regions:
            best_idx = None
            best_iou = 0.0
            for p_idx, pred_region in enumerate(pred_regions):
                if p_idx in matched_pred:
                    continue
                if pred_region["page"] != gt_region["page"]:
                    continue
                if pred_region["kind"] != gt_region["kind"]:
                    continue
                cur_iou = _iou(gt_region["bbox"], pred_region["bbox"])
                if cur_iou > best_iou:
                    best_iou = cur_iou
                    best_idx = p_idx
            if best_idx is not None and best_iou >= iou_threshold:
                matched_pred.add(best_idx)
                doc_tp += 1
        doc_fp = max(0, len(pred_regions) - len(matched_pred))
        doc_fn = max(0, len(gt_regions) - doc_tp)
        region_tp += doc_tp
        region_fp += doc_fp
        region_fn += doc_fn

        # Problem text metrics
        gt_problems = _problem_map(gt_doc)
        pred_problems = _problem_map(pred_doc)
        for uid, gt_problem in gt_problems.items():
            gt_text = str(gt_problem.get("description_text", "") or "")
            pred_text = str(pred_problems.get(uid, {}).get("description_text", "") or "")
            dist = levenshtein_distance(gt_text, pred_text)
            text_distance_sum += dist
            text_char_total += max(1, len(gt_text))
            denom = max(len(gt_text), len(pred_text), 1)
            text_norm_sum += dist / float(denom)
            text_pair_count += 1

        # Figure association metrics
        gt_figure_pairs = _figure_pairs(gt_problems)
        pred_figure_pairs = _figure_pairs(pred_problems)
        local_tp = len(gt_figure_pairs & pred_figure_pairs)
        local_fp = len(pred_figure_pairs - gt_figure_pairs)
        local_fn = len(gt_figure_pairs - pred_figure_pairs)
        figure_tp += local_tp
        figure_fp += local_fp
        figure_fn += local_fn

        # Template matching
        gt_template = str(gt_doc.get("template_version_id", "") or "")
        pred_template = str(pred_doc.get("template_version_id", "") or "")
        if gt_template:
            template_total += 1
            if pred_template == gt_template:
                template_correct += 1

        # Error-tag metrics
        gt_tags = _extract_error_tag_tuples(gt_doc)
        pred_tags = _extract_error_tag_tuples(pred_doc)
        tag_tp_local = len(gt_tags & pred_tags)
        tag_fp_local = len(pred_tags - gt_tags)
        tag_fn_local = len(gt_tags - pred_tags)
        tag_tp += tag_tp_local
        tag_fp += tag_fp_local
        tag_fn += tag_fn_local

        tags_in_doc = sorted({tag for _p, _c, tag in gt_tags | pred_tags})
        for tag in tags_in_doc:
            gt_has = any(cur_tag == tag for _p, _c, cur_tag in gt_tags)
            pred_has = any(cur_tag == tag for _p, _c, cur_tag in pred_tags)
            if gt_has and pred_has:
                per_tag_counts[tag]["tp"] += 1
            elif pred_has and not gt_has:
                per_tag_counts[tag]["fp"] += 1
            elif gt_has and not pred_has:
                per_tag_counts[tag]["fn"] += 1

        # Rubric-score metrics
        gt_points = _extract_rubric_points(gt_doc)
        pred_points = _extract_rubric_points(pred_doc)
        for key, gt_value in gt_points.items():
            pred_value = pred_points.get(key, 0.0)
            diff = abs(float(gt_value) - float(pred_value))
            rubric_abs_errors.append(diff)
            rubric_total += 1
            if diff == 0.0:
                rubric_exact += 1
            qwk_true.append(float(gt_value))
            qwk_pred.append(float(pred_value))

        # Feedback-dimension agreement
        gt_feedback = _extract_feedback_dimensions(gt_doc)
        pred_feedback = _extract_feedback_dimensions(pred_doc)
        for problem_uid, gt_dims in gt_feedback.items():
            pred_dims = pred_feedback.get(problem_uid)
            if pred_dims is None:
                continue
            for dim in ("correctness", "specificity", "actionability"):
                feedback_dim_total[dim] += 1
                if int(pred_dims.get(dim, 0)) == int(gt_dims.get(dim, 0)):
                    feedback_dim_matches[dim] += 1

        per_doc_rows.append(
            {
                "doc_id": doc_id,
                "region_tp": doc_tp,
                "region_fp": doc_fp,
                "region_fn": doc_fn,
                "template_gt": gt_template,
                "template_pred": pred_template,
                "template_correct": bool(gt_template and pred_template == gt_template),
            }
        )

    def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        if precision + recall <= 0.0:
            f1 = 0.0
        else:
            f1 = float(2.0 * precision * recall / (precision + recall))
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    region_scores = _prf(region_tp, region_fp, region_fn)
    figure_scores = _prf(figure_tp, figure_fp, figure_fn)
    tag_scores = _prf(tag_tp, tag_fp, tag_fn)

    macro_f1_values: list[float] = []
    for counters in per_tag_counts.values():
        macro_f1_values.append(
            _prf(
                int(counters.get("tp", 0)),
                int(counters.get("fp", 0)),
                int(counters.get("fn", 0)),
            )["f1"]
        )
    macro_tag_f1 = float(sum(macro_f1_values) / len(macro_f1_values)) if macro_f1_values else 0.0

    cer = float(text_distance_sum / text_char_total) if text_char_total > 0 else 0.0
    norm_edit = float(text_norm_sum / text_pair_count) if text_pair_count > 0 else 0.0
    template_acc = float(template_correct / template_total) if template_total > 0 else 0.0

    rubric_mae = float(sum(rubric_abs_errors) / len(rubric_abs_errors)) if rubric_abs_errors else 0.0
    rubric_exact_match = float(rubric_exact / rubric_total) if rubric_total > 0 else 0.0
    rubric_qwk = float(quadratic_weighted_kappa(qwk_true, qwk_pred)) if rubric_total > 0 else 0.0

    feedback_scores: dict[str, float] = {}
    for dim in ("correctness", "specificity", "actionability"):
        total = int(feedback_dim_total.get(dim, 0))
        matches = int(feedback_dim_matches.get(dim, 0))
        feedback_scores[f"{dim}_agreement"] = float(matches / total) if total > 0 else 0.0
    feedback_scores["overall_agreement"] = (
        float(
            sum(feedback_dim_matches.values()) / sum(feedback_dim_total.values())
        )
        if sum(feedback_dim_total.values()) > 0
        else 0.0
    )

    return {
        "doc_count": len(docs),
        "region_detection": {
            **region_scores,
            "iou_threshold": float(iou_threshold),
            "tp": region_tp,
            "fp": region_fp,
            "fn": region_fn,
        },
        "description_transcription": {
            "cer": cer,
            "normalized_edit_distance": norm_edit,
            "pairs": text_pair_count,
        },
        "figure_association": {
            **figure_scores,
            "tp": figure_tp,
            "fp": figure_fp,
            "fn": figure_fn,
        },
        "template_matching": {
            "top1_accuracy": template_acc,
            "correct": template_correct,
            "total": template_total,
        },
        "error_tagging": {
            "micro_precision": tag_scores["precision"],
            "micro_recall": tag_scores["recall"],
            "micro_f1": tag_scores["f1"],
            "macro_f1": macro_tag_f1,
            "tp": tag_tp,
            "fp": tag_fp,
            "fn": tag_fn,
        },
        "rubric_scoring": {
            "mae": rubric_mae,
            "exact_match_rate": rubric_exact_match,
            "qwk": rubric_qwk,
            "pairs": rubric_total,
        },
        "feedback_quality": feedback_scores,
        "per_doc": per_doc_rows,
    }
