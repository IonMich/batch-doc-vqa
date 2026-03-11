#!/usr/bin/env python3
"""Local UI for TA benchmark annotation and verification."""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from datetime import datetime, timezone
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import parse_qs, unquote, urlparse

from rich.console import Console
from rich.table import Table

from ..ta_benchmark.constants import (
    DEFAULT_TA_SCHEMA_PATH,
    DEFAULT_TA_TAXONOMY_PATH,
    PII_WARNING_BANNER,
    TA_SCHEMA_VERSION,
)
from ..ta_benchmark.io_utils import load_doc_info, load_test_ids, read_json
from ..ta_benchmark.schema import load_error_taxonomy, load_ta_schema, validate_label_payload

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


console = Console()


def _read_json(path: Optional[str | Path]) -> Any:
    if not path:
        return None
    return read_json(path)


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _build_default_label(
    *,
    doc_id: int,
    assignment_id: int,
    submission_id: str,
    proposal: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    label = {
        "schema_version": TA_SCHEMA_VERSION,
        "doc_id": int(doc_id),
        "submission_id": str(submission_id or f"unknown-{doc_id}"),
        "assignment_id": int(max(1, assignment_id)),
        "template_version_id": "",
        "evidence_regions": [],
        "problems": [],
        "tier2": {
            "rubric_scores": [],
            "feedback": [],
        },
        "review": {
            "annotator_id": "",
            "status": "draft",
            "updated_at_utc": _now_utc(),
            "proposal_action": "",
        },
    }
    if isinstance(proposal, dict):
        for key in ("template_version_id", "evidence_regions", "problems", "tier2"):
            if key in proposal:
                label[key] = proposal[key]
        if isinstance(proposal.get("review"), dict):
            label["review"] = {
                **label["review"],
                **proposal["review"],
            }
    return label


def _load_doc_index(
    *,
    doc_info_csv: Path,
    test_ids_csv: Optional[Path],
    images_dir: Path,
) -> dict[int, dict[str, Any]]:
    doc_info = load_doc_info(doc_info_csv)
    gt = load_test_ids(test_ids_csv) if test_ids_csv else {}

    docs: dict[int, dict[str, Any]] = {}
    for doc_id in sorted(doc_info):
        pages = []
        for row in doc_info[doc_id]:
            filename = str(row.get("filename", "")).strip()
            path_obj = Path(filename)
            if not path_obj.is_absolute():
                path_obj = (images_dir / path_obj).resolve(strict=False)
            pages.append(
                {
                    "page": _parse_int(row.get("page"), 0),
                    "filename": filename,
                    "resolved_path": str(path_obj),
                    "image_url": f"/img/{quote_path(filename)}",
                }
            )
        pages.sort(key=lambda item: int(item["page"]))
        gt_row = gt.get(doc_id, {})
        assignment_id = _parse_int(gt_row.get("assignment_id"), 1)
        submission_id = str(gt_row.get("submission_id", "") or "").strip() or f"unknown-{doc_id}"
        docs[doc_id] = {
            "doc_id": doc_id,
            "pages": pages,
            "ground_truth": gt_row,
            "assignment_id": max(1, assignment_id),
            "submission_id": submission_id,
        }
    return docs


def _load_labels_map(labels_dir: Path) -> dict[int, dict[str, Any]]:
    labels: dict[int, dict[str, Any]] = {}
    docs_dir = labels_dir / "docs"
    if not docs_dir.exists():
        return labels
    for path in sorted(docs_dir.glob("doc-*.json")):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        doc_id = _parse_int(payload.get("doc_id"), -1)
        if doc_id < 0:
            continue
        labels[doc_id] = payload
    return labels


def _load_proposals_map(path: Optional[str | Path]) -> dict[int, dict[str, Any]]:
    payload = _read_json(path)
    if not isinstance(payload, dict):
        return {}
    docs = payload.get("docs", [])
    if not isinstance(docs, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for row in docs:
        if not isinstance(row, dict):
            continue
        doc_id = _parse_int(row.get("doc_id"), -1)
        if doc_id < 0:
            continue
        result[doc_id] = row
    return result


def _load_rubrics(rubrics_dir: Optional[Path]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    if rubrics_dir is None or not rubrics_dir.exists():
        return result

    yaml_paths = sorted(list(rubrics_dir.glob("*.yaml")) + list(rubrics_dir.glob("*.yml")))
    if yaml_paths and yaml is None:
        console.print(
            "[yellow]PyYAML unavailable; skipping YAML rubrics. Use JSON rubrics (*.json) or install PyYAML.[/yellow]"
        )

    for path in sorted(list(rubrics_dir.glob("*.json")) + yaml_paths):
        try:
            if path.suffix.lower() == ".json":
                payload = read_json(path)
            else:
                if yaml is None:
                    continue
                with open(path, "r", encoding="utf-8") as f:
                    payload = yaml.safe_load(f)
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        assignment_id = payload.get("assignment_id")
        if assignment_id is None:
            stem = path.stem.strip()
            if stem.isdigit():
                assignment_id = int(stem)
        if assignment_id is None:
            continue
        criteria = payload.get("criteria", [])
        if not isinstance(criteria, list):
            continue
        cleaned: list[dict[str, Any]] = []
        for row in criteria:
            if not isinstance(row, dict):
                continue
            criterion_id = str(row.get("criterion_id", "")).strip()
            label = str(row.get("label", "")).strip()
            max_points = row.get("max_points")
            if not criterion_id or not label:
                continue
            try:
                max_points_f = float(max_points)
            except (TypeError, ValueError):
                max_points_f = 0.0
            cleaned.append(
                {
                    "criterion_id": criterion_id,
                    "label": label,
                    "max_points": max_points_f,
                    "description": str(row.get("description", "") or "").strip(),
                }
            )
        result[str(int(assignment_id))] = cleaned
    return result


def quote_path(raw: str) -> str:
    return str(raw).replace("%", "%25").replace("/", "%2F").replace("\\", "%5C")


def unquote_path(raw: str) -> str:
    return unquote(str(raw)).replace("\\", "/")


def save_label(
    *,
    labels_dir: Path,
    payload: dict[str, Any],
    schema: dict[str, Any],
    known_error_tags: set[str],
) -> tuple[bool, str, list[dict[str, Any]]]:
    doc_id = _parse_int(payload.get("doc_id"), -1)
    if doc_id < 0:
        return False, "", [{"code": "missing_doc_id", "message": "doc_id is required", "path": "$.doc_id"}]

    payload = dict(payload)
    review = payload.get("review", {})
    if not isinstance(review, dict):
        review = {}
    review["updated_at_utc"] = _now_utc()
    payload["review"] = review

    result = validate_label_payload(
        payload,
        schema=schema,
        known_error_tags=known_error_tags,
    )
    if not result.is_valid:
        return (
            False,
            "",
            [
                {
                    "level": issue.level,
                    "code": issue.code,
                    "message": issue.message,
                    "path": issue.path,
                }
                for issue in result.issues
            ],
        )

    docs_dir = labels_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    out_path = docs_dir / f"doc-{doc_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return True, str(out_path), []


def build_html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>TA Annotation UI</title>
  <style>
    :root {
      --bg: #f3f2ef;
      --panel: #fff;
      --ink: #1f2328;
      --muted: #6a737d;
      --line: #d0d7de;
      --accent: #1f7a8c;
      --warn: #9a5a00;
      --bad: #b42318;
      --ok: #1f7a1f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
      background: radial-gradient(circle at top right, #ece7d9 0%, var(--bg) 70%);
      color: var(--ink);
      height: 100vh;
      overflow: hidden;
    }
    .app {
      height: 100vh;
      display: grid;
      grid-template-rows: auto minmax(0, 1fr);
      gap: 8px;
      padding: 10px;
    }
    .topbar {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      display: grid;
      grid-template-columns: auto auto auto auto auto auto minmax(140px, 200px) minmax(240px, 1fr);
      gap: 8px;
      align-items: center;
    }
    button, select, input, textarea {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 7px 9px;
      font-size: 13px;
      background: #fff;
    }
    textarea { min-height: 48px; font-family: inherit; }
    button { cursor: pointer; font-weight: 600; }
    .status {
      font-size: 12px;
      color: var(--muted);
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
    }
    .workspace {
      min-height: 0;
      display: grid;
      grid-template-columns: minmax(220px, 280px) minmax(0, 1fr) minmax(320px, 420px);
      gap: 8px;
    }
    .panel {
      min-height: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px;
      display: flex;
      flex-direction: column;
      gap: 8px;
    }
    .panel h3 {
      margin: 0;
      font-size: 14px;
    }
    .doc-list {
      min-height: 0;
      overflow: auto;
      display: grid;
      gap: 6px;
    }
    .doc-item {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 6px;
      font-size: 12px;
      background: #fff;
      cursor: pointer;
    }
    .doc-item.active {
      border-color: var(--accent);
      box-shadow: 0 0 0 2px rgba(31, 122, 140, 0.15);
    }
    .doc-item .title { font-weight: 700; }
    .canvas-wrap {
      position: relative;
      min-height: 0;
      flex: 1;
      overflow: auto;
      background: #f8f9fa;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      display: flex;
      justify-content: center;
      align-items: flex-start;
    }
    .image-stack {
      position: relative;
      display: inline-block;
    }
    #pageImage {
      display: block;
      max-width: min(100%, 960px);
      border-radius: 6px;
      background: #f0f0f0;
    }
    #overlayCanvas {
      position: absolute;
      top: 0;
      left: 0;
      cursor: crosshair;
    }
    .side {
      min-height: 0;
      overflow: auto;
      display: grid;
      gap: 10px;
      align-content: start;
      padding-right: 4px;
    }
    .block {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px;
      background: #fcfcfa;
    }
    .row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 6px;
      margin-bottom: 6px;
    }
    .row3 {
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 6px;
      margin-bottom: 6px;
    }
    .mini {
      font-size: 12px;
      color: var(--muted);
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 4px;
      vertical-align: top;
    }
    th { background: #f7f6f3; }
    .pill {
      font-size: 11px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      display: inline-block;
      margin-right: 4px;
      background: #f4f4f2;
    }
    .ok { color: var(--ok); }
    .warn { color: var(--warn); }
    .bad { color: var(--bad); }
    .hint {
      font-size: 12px;
      color: var(--muted);
      margin-top: 4px;
      line-height: 1.3;
    }
    @media (max-width: 1200px) {
      .workspace { grid-template-columns: 1fr; }
      body { overflow: auto; height: auto; }
      .app { height: auto; }
      .topbar { grid-template-columns: repeat(4, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <div class="app">
    <section class="topbar">
      <button id="prevBtn" type="button">Prev Doc</button>
      <button id="nextBtn" type="button">Next Doc</button>
      <button id="saveBtn" type="button">Save</button>
      <button id="drawBtn" type="button">Add Box (B)</button>
      <select id="pageSelect"></select>
      <input id="jumpInput" type="number" min="0" step="1" placeholder="Doc id" />
      <button id="jumpBtn" type="button">Jump</button>
      <div id="status" class="status">Loading...</div>
    </section>

    <section class="workspace">
      <aside class="panel">
        <h3>Docs</h3>
        <div id="docList" class="doc-list"></div>
      </aside>

      <main class="panel">
        <h3>Page Viewer</h3>
        <div class="hint">Shortcuts: <span class="pill">Left/Right</span><span class="pill">Cmd/Ctrl+S</span><span class="pill">B</span></div>
        <div class="canvas-wrap">
          <div class="image-stack">
            <img id="pageImage" alt="Document page" />
            <canvas id="overlayCanvas"></canvas>
          </div>
        </div>
      </main>

      <aside class="panel">
        <h3>Annotation</h3>
        <div class="side">
          <div class="block">
            <div class="row">
              <label>Template Version
                <input id="templateVersionInput" type="text" />
              </label>
              <label>Review Status
                <select id="reviewStatusSelect">
                  <option value="draft">draft</option>
                  <option value="in_review">in_review</option>
                  <option value="verified">verified</option>
                </select>
              </label>
            </div>
            <div class="row">
              <label>Annotator ID
                <input id="annotatorInput" type="text" />
              </label>
              <label>Proposal Action
                <select id="proposalActionSelect">
                  <option value="">(unset)</option>
                  <option value="accepted">accepted</option>
                  <option value="edited">edited</option>
                  <option value="rejected">rejected</option>
                </select>
              </label>
            </div>
            <div id="proposalHint" class="hint"></div>
          </div>

          <div class="block">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <h3 style="margin:0;">Evidence Regions</h3>
              <div>
                <button id="addEvidenceBtn" type="button">Add Row</button>
                <button id="delEvidenceBtn" type="button">Delete Selected</button>
              </div>
            </div>
            <table id="evidenceTable">
              <thead>
                <tr>
                  <th>ID</th><th>Kind</th><th>Page</th><th>BBox</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>

          <div class="block">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <h3 style="margin:0;">Problems</h3>
              <button id="addProblemBtn" type="button">Add Problem</button>
            </div>
            <table id="problemTable">
              <thead>
                <tr>
                  <th>UID</th><th>Number</th><th>Description</th><th>Description refs</th><th>Figure refs</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>

          <div class="block">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <h3 style="margin:0;">Rubric Scores</h3>
              <div>
                <button id="loadRubricBtn" type="button">Load Rubric</button>
                <button id="addRubricBtn" type="button">Add Score</button>
              </div>
            </div>
            <table id="rubricTable">
              <thead>
                <tr>
                  <th>Problem</th><th>Criterion</th><th>Max</th><th>Awarded</th><th>Error tags</th><th>Evidence refs</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>

          <div class="block">
            <div style="display:flex;justify-content:space-between;align-items:center;">
              <h3 style="margin:0;">Feedback</h3>
              <button id="addFeedbackBtn" type="button">Add Feedback</button>
            </div>
            <table id="feedbackTable">
              <thead>
                <tr>
                  <th>Problem</th><th>Comment</th><th>Spec</th><th>Action</th><th>Correct</th><th>Evidence refs</th>
                </tr>
              </thead>
              <tbody></tbody>
            </table>
          </div>
        </div>
      </aside>
    </section>
  </div>

  <script>
    const state = {
      docs: [],
      labels: {},
      proposals: {},
      rubrics: {},
      index: 0,
      page: 0,
      selectedEvidenceIndex: -1,
      drawMode: false,
      drawStart: null,
      drawPreview: null,
      dirty: false,
      autosaveMs: 15000,
    };

    function byId(id) { return document.getElementById(id); }
    function esc(raw) {
      return String(raw ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;");
    }
    function parseCsv(raw) {
      return String(raw || "").split(",").map((x) => x.trim()).filter(Boolean);
    }
    function toCsv(items) {
      return (Array.isArray(items) ? items : []).map((x) => String(x || "").trim()).filter(Boolean).join(", ");
    }
    function nowIso() { return new Date().toISOString(); }

    function currentDoc() {
      return state.docs[state.index] || null;
    }

    function currentDocId() {
      const doc = currentDoc();
      return doc ? Number(doc.doc_id) : -1;
    }

    function activeLabel() {
      const docId = currentDocId();
      if (docId < 0) return null;
      return state.labels[String(docId)] || state.labels[docId] || null;
    }

    function setDirty(val = true) {
      state.dirty = Boolean(val);
      const status = byId("status");
      if (state.dirty) status.innerHTML = `<span class="warn">Unsaved changes</span>`;
    }

    function nextEvidenceId(label) {
      const used = new Set((label.evidence_regions || []).map((row) => String(row.evidence_id || "").trim()));
      for (let i = 1; i < 5000; i += 1) {
        const candidate = `ev${i}`;
        if (!used.has(candidate)) return candidate;
      }
      return `ev${Date.now()}`;
    }

    function ensureLabel() {
      const doc = currentDoc();
      if (!doc) return null;
      const key = String(doc.doc_id);
      let label = state.labels[key] || state.labels[doc.doc_id];
      if (!label) return null;
      if (!Array.isArray(label.evidence_regions)) label.evidence_regions = [];
      if (!Array.isArray(label.problems)) label.problems = [];
      if (!label.tier2 || typeof label.tier2 !== "object") label.tier2 = {};
      if (!Array.isArray(label.tier2.rubric_scores)) label.tier2.rubric_scores = [];
      if (!Array.isArray(label.tier2.feedback)) label.tier2.feedback = [];
      if (!label.review || typeof label.review !== "object") label.review = {};
      state.labels[key] = label;
      return label;
    }

    function drawOverlay() {
      const canvas = byId("overlayCanvas");
      const image = byId("pageImage");
      const label = ensureLabel();
      if (!canvas || !image || !label) return;

      const width = image.clientWidth;
      const height = image.clientHeight;
      canvas.width = width;
      canvas.height = height;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, width, height);

      const page = Number(state.page);
      (label.evidence_regions || []).forEach((row, idx) => {
        if (Number(row.page) !== page) return;
        const bbox = Array.isArray(row.bbox) ? row.bbox : [0, 0, 0, 0];
        if (bbox.length !== 4) return;
        const [x1, y1, x2, y2] = bbox.map((v) => Number(v || 0));
        const rx = x1 * width;
        const ry = y1 * height;
        const rw = (x2 - x1) * width;
        const rh = (y2 - y1) * height;
        const selected = idx === state.selectedEvidenceIndex;
        ctx.lineWidth = selected ? 3 : 2;
        ctx.strokeStyle = selected ? "#b42318" : "#1f7a8c";
        ctx.strokeRect(rx, ry, rw, rh);
        ctx.fillStyle = selected ? "rgba(180,35,24,0.18)" : "rgba(31,122,140,0.12)";
        ctx.fillRect(rx, ry, rw, rh);
        const labelText = `${row.evidence_id || "?"} (${row.kind || "other"})`;
        ctx.fillStyle = "#111";
        ctx.font = "12px IBM Plex Sans";
        ctx.fillText(labelText, rx + 4, Math.max(12, ry + 12));
      });

      if (state.drawPreview) {
        const p = state.drawPreview;
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#9a5a00";
        ctx.strokeRect(p.x, p.y, p.w, p.h);
        ctx.fillStyle = "rgba(154,90,0,0.15)";
        ctx.fillRect(p.x, p.y, p.w, p.h);
      }
    }

    function renderDocList() {
      const container = byId("docList");
      const html = state.docs.map((doc, idx) => {
        const key = String(doc.doc_id);
        const label = state.labels[key] || state.labels[doc.doc_id] || {};
        const status = String(label.review?.status || "draft");
        const proposal = state.proposals[key] || state.proposals[doc.doc_id];
        const hasProposal = Boolean(proposal);
        return `
          <button class="doc-item ${idx === state.index ? "active" : ""}" data-idx="${idx}">
            <div class="title">Doc ${doc.doc_id}</div>
            <div>Assignment ${doc.assignment_id} · ${status}</div>
            <div>${hasProposal ? '<span class="pill">proposal</span>' : ''}</div>
          </button>`;
      }).join("");
      container.innerHTML = html;
      container.querySelectorAll(".doc-item").forEach((el) => {
        el.addEventListener("click", () => {
          state.index = Number(el.getAttribute("data-idx") || 0);
          state.page = Number((currentDoc()?.pages?.[0]?.page) || 1);
          state.selectedEvidenceIndex = -1;
          localStorage.setItem("taAnnotDocId", String(currentDocId()));
          renderAll();
        });
      });
    }

    function renderPageSelect() {
      const select = byId("pageSelect");
      const doc = currentDoc();
      if (!doc) {
        select.innerHTML = "";
        return;
      }
      select.innerHTML = (doc.pages || []).map((row) => {
        const p = Number(row.page);
        return `<option value="${p}" ${p === Number(state.page) ? "selected" : ""}>Page ${p}</option>`;
      }).join("");
    }

    function setImage() {
      const doc = currentDoc();
      if (!doc) return;
      const pageRow = (doc.pages || []).find((row) => Number(row.page) === Number(state.page)) || doc.pages?.[0];
      if (!pageRow) return;
      const image = byId("pageImage");
      image.onload = () => drawOverlay();
      image.src = pageRow.image_url;
    }

    function renderMeta() {
      const label = ensureLabel();
      const doc = currentDoc();
      if (!label || !doc) return;
      byId("templateVersionInput").value = String(label.template_version_id || "");
      byId("reviewStatusSelect").value = String(label.review?.status || "draft");
      byId("annotatorInput").value = String(label.review?.annotator_id || "");
      byId("proposalActionSelect").value = String(label.review?.proposal_action || "");

      const proposal = state.proposals[String(doc.doc_id)] || state.proposals[doc.doc_id];
      if (proposal && proposal.proposal?.confidence_hints) {
        const hints = proposal.proposal.confidence_hints;
        byId("proposalHint").innerHTML =
          `Proposal hints: identity completeness=${Number(hints.identity_completeness || 0).toFixed(2)}, `
          + `missing_name=${Boolean(hints.missing_name)}, missing_id=${Boolean(hints.missing_id)}, missing_section=${Boolean(hints.missing_section)}`;
      } else {
        byId("proposalHint").innerHTML = "No proposal loaded for this doc.";
      }
    }

    function rowInput(value, cls) {
      return `<input class="${cls}" type="text" value="${esc(value)}" />`;
    }

    function renderEvidenceTable() {
      const body = byId("evidenceTable").querySelector("tbody");
      const label = ensureLabel();
      const rows = (label?.evidence_regions || []);
      body.innerHTML = rows.map((row, idx) => {
        const bbox = Array.isArray(row.bbox) && row.bbox.length === 4 ? row.bbox : [0, 0, 0, 0];
        return `
          <tr data-idx="${idx}" class="${idx === state.selectedEvidenceIndex ? "selected" : ""}">
            <td>${rowInput(row.evidence_id || "", "ev-id")}</td>
            <td>${rowInput(row.kind || "other", "ev-kind")}</td>
            <td>${rowInput(row.page || state.page, "ev-page")}</td>
            <td>${rowInput(bbox.join(","), "ev-bbox")}</td>
          </tr>`;
      }).join("");
      body.querySelectorAll("tr").forEach((tr) => {
        tr.addEventListener("click", () => {
          state.selectedEvidenceIndex = Number(tr.getAttribute("data-idx") || -1);
          drawOverlay();
        });
      });
      body.querySelectorAll("input").forEach((input) => {
        input.addEventListener("change", syncEvidenceFromTable);
      });
    }

    function syncEvidenceFromTable() {
      const label = ensureLabel();
      if (!label) return;
      const rows = Array.from(byId("evidenceTable").querySelectorAll("tbody tr"));
      label.evidence_regions = rows.map((tr) => {
        const get = (cls) => tr.querySelector(`.${cls}`)?.value || "";
        const bboxVals = String(get("ev-bbox")).split(",").map((v) => Number(v.trim() || 0));
        while (bboxVals.length < 4) bboxVals.push(0);
        return {
          evidence_id: String(get("ev-id")).trim() || nextEvidenceId(label),
          kind: String(get("ev-kind")).trim() || "other",
          page: Number(get("ev-page")) || Number(state.page),
          bbox: bboxVals.slice(0, 4).map((v) => Math.max(0, Math.min(1, Number(v)))),
        };
      });
      setDirty(true);
      drawOverlay();
    }

    function renderProblemTable() {
      const label = ensureLabel();
      const body = byId("problemTable").querySelector("tbody");
      const rows = (label?.problems || []);
      body.innerHTML = rows.map((row) => `
        <tr>
          <td>${rowInput(row.problem_uid || "", "pr-uid")}</td>
          <td>${rowInput(row.problem_number || "", "pr-number")}</td>
          <td><textarea class="pr-desc">${esc(row.description_text || "")}</textarea></td>
          <td>${rowInput(toCsv(row.description_evidence_ids || []), "pr-desc-refs")}</td>
          <td>${rowInput(toCsv(row.figure_evidence_ids || []), "pr-fig-refs")}</td>
        </tr>
      `).join("");
      body.querySelectorAll("input,textarea").forEach((input) => input.addEventListener("change", syncProblemsFromTable));
    }

    function syncProblemsFromTable() {
      const label = ensureLabel();
      if (!label) return;
      const rows = Array.from(byId("problemTable").querySelectorAll("tbody tr"));
      label.problems = rows.map((tr, idx) => ({
        problem_uid: String(tr.querySelector(".pr-uid")?.value || "").trim() || `p${idx + 1}`,
        problem_number: String(tr.querySelector(".pr-number")?.value || "").trim(),
        description_text: String(tr.querySelector(".pr-desc")?.value || "").trim(),
        description_evidence_ids: parseCsv(tr.querySelector(".pr-desc-refs")?.value || ""),
        figure_evidence_ids: parseCsv(tr.querySelector(".pr-fig-refs")?.value || ""),
      }));
      setDirty(true);
    }

    function renderRubricTable() {
      const label = ensureLabel();
      const body = byId("rubricTable").querySelector("tbody");
      const rows = (label?.tier2?.rubric_scores || []);
      body.innerHTML = rows.map((row) => `
        <tr>
          <td>${rowInput(row.problem_uid || "", "rb-problem")}</td>
          <td>${rowInput(row.criterion_id || "", "rb-criterion")}</td>
          <td>${rowInput(row.max_points ?? 0, "rb-max")}</td>
          <td>${rowInput(row.awarded_points ?? 0, "rb-awarded")}</td>
          <td>${rowInput(toCsv(row.error_tags || []), "rb-tags")}</td>
          <td>${rowInput(toCsv(row.evidence_ids || []), "rb-refs")}</td>
        </tr>
      `).join("");
      body.querySelectorAll("input").forEach((input) => input.addEventListener("change", syncRubricFromTable));
    }

    function syncRubricFromTable() {
      const label = ensureLabel();
      if (!label) return;
      const rows = Array.from(byId("rubricTable").querySelectorAll("tbody tr"));
      label.tier2.rubric_scores = rows.map((tr) => ({
        problem_uid: String(tr.querySelector(".rb-problem")?.value || "").trim(),
        criterion_id: String(tr.querySelector(".rb-criterion")?.value || "").trim(),
        max_points: Number(tr.querySelector(".rb-max")?.value || 0),
        awarded_points: Number(tr.querySelector(".rb-awarded")?.value || 0),
        error_tags: parseCsv(tr.querySelector(".rb-tags")?.value || ""),
        evidence_ids: parseCsv(tr.querySelector(".rb-refs")?.value || ""),
      })).filter((row) => row.problem_uid && row.criterion_id);
      setDirty(true);
    }

    function renderFeedbackTable() {
      const label = ensureLabel();
      const body = byId("feedbackTable").querySelector("tbody");
      const rows = (label?.tier2?.feedback || []);
      body.innerHTML = rows.map((row) => `
        <tr>
          <td>${rowInput(row.problem_uid || "", "fb-problem")}</td>
          <td><textarea class="fb-comment">${esc(row.comment || "")}</textarea></td>
          <td>${rowInput(row.specificity ?? 0, "fb-spec")}</td>
          <td>${rowInput(row.actionability ?? 0, "fb-action")}</td>
          <td>${rowInput(row.correctness ?? 1, "fb-correct")}</td>
          <td>${rowInput(toCsv(row.evidence_ids || []), "fb-refs")}</td>
        </tr>
      `).join("");
      body.querySelectorAll("input,textarea").forEach((input) => input.addEventListener("change", syncFeedbackFromTable));
    }

    function syncFeedbackFromTable() {
      const label = ensureLabel();
      if (!label) return;
      const rows = Array.from(byId("feedbackTable").querySelectorAll("tbody tr"));
      label.tier2.feedback = rows.map((tr) => ({
        problem_uid: String(tr.querySelector(".fb-problem")?.value || "").trim(),
        comment: String(tr.querySelector(".fb-comment")?.value || "").trim(),
        specificity: Number(tr.querySelector(".fb-spec")?.value || 0),
        actionability: Number(tr.querySelector(".fb-action")?.value || 0),
        correctness: Number(tr.querySelector(".fb-correct")?.value || 1),
        evidence_ids: parseCsv(tr.querySelector(".fb-refs")?.value || ""),
      })).filter((row) => row.problem_uid);
      setDirty(true);
    }

    function syncMetaFromInputs() {
      const label = ensureLabel();
      if (!label) return;
      label.template_version_id = String(byId("templateVersionInput").value || "").trim();
      label.review.status = String(byId("reviewStatusSelect").value || "draft");
      label.review.annotator_id = String(byId("annotatorInput").value || "").trim();
      label.review.proposal_action = String(byId("proposalActionSelect").value || "");
      setDirty(true);
    }

    function applyRubricPreset() {
      const doc = currentDoc();
      const label = ensureLabel();
      if (!doc || !label) return;
      const rubric = state.rubrics[String(doc.assignment_id)] || [];
      if (!Array.isArray(rubric) || rubric.length === 0) {
        byId("status").innerHTML = `<span class="warn">No rubric found for assignment ${doc.assignment_id}</span>`;
        return;
      }
      if (!Array.isArray(label.tier2.rubric_scores)) label.tier2.rubric_scores = [];
      if (label.tier2.rubric_scores.length > 0) {
        byId("status").innerHTML = `<span class="warn">Rubric rows already exist; add manually or clear first.</span>`;
        return;
      }
      const problemUid = String((label.problems?.[0]?.problem_uid) || "p1");
      label.tier2.rubric_scores = rubric.map((row) => ({
        problem_uid: problemUid,
        criterion_id: String(row.criterion_id || ""),
        max_points: Number(row.max_points || 0),
        awarded_points: Number(row.max_points || 0),
        error_tags: [],
        evidence_ids: [],
      }));
      setDirty(true);
      renderRubricTable();
    }

    function renderAll() {
      const doc = currentDoc();
      if (!doc) return;
      renderDocList();
      renderPageSelect();
      renderMeta();
      renderEvidenceTable();
      renderProblemTable();
      renderRubricTable();
      renderFeedbackTable();
      setImage();
      byId("jumpInput").value = String(doc.doc_id);
      if (!state.dirty) {
        byId("status").innerHTML = `Doc ${doc.doc_id} · assignment ${doc.assignment_id} · page ${state.page}`;
      }
    }

    function getCurrentLabelPayload() {
      const label = ensureLabel();
      if (!label) return null;
      syncMetaFromInputs();
      syncEvidenceFromTable();
      syncProblemsFromTable();
      syncRubricFromTable();
      syncFeedbackFromTable();
      return label;
    }

    async function saveCurrentLabel({silent = false} = {}) {
      const doc = currentDoc();
      const label = getCurrentLabelPayload();
      if (!doc || !label) return false;
      label.review.updated_at_utc = nowIso();
      const response = await fetch("/save", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({doc_id: doc.doc_id, label}),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        const issues = Array.isArray(payload.issues) ? payload.issues : [];
        const first = issues[0]?.message || payload.message || "Save failed";
        byId("status").innerHTML = `<span class="bad">${esc(first)}</span>`;
        return false;
      }
      state.labels[String(doc.doc_id)] = payload.label || label;
      state.dirty = false;
      if (!silent) {
        byId("status").innerHTML = `<span class="ok">Saved ${esc(payload.path || "")}</span>`;
      }
      renderDocList();
      return true;
    }

    function gotoDocByIndex(nextIndex) {
      const bounded = Math.max(0, Math.min(state.docs.length - 1, Number(nextIndex)));
      if (bounded === state.index) return;
      state.index = bounded;
      state.page = Number(currentDoc()?.pages?.[0]?.page || 1);
      state.selectedEvidenceIndex = -1;
      localStorage.setItem("taAnnotDocId", String(currentDocId()));
      renderAll();
    }

    function setupCanvasInteractions() {
      const canvas = byId("overlayCanvas");
      const image = byId("pageImage");
      canvas.addEventListener("mousedown", (ev) => {
        if (!state.drawMode) return;
        const rect = canvas.getBoundingClientRect();
        state.drawStart = {x: ev.clientX - rect.left, y: ev.clientY - rect.top};
        state.drawPreview = null;
      });
      canvas.addEventListener("mousemove", (ev) => {
        if (!state.drawMode || !state.drawStart) return;
        const rect = canvas.getBoundingClientRect();
        const x = ev.clientX - rect.left;
        const y = ev.clientY - rect.top;
        const sx = state.drawStart.x;
        const sy = state.drawStart.y;
        state.drawPreview = {
          x: Math.min(sx, x),
          y: Math.min(sy, y),
          w: Math.abs(x - sx),
          h: Math.abs(y - sy),
        };
        drawOverlay();
      });
      canvas.addEventListener("mouseup", () => {
        if (!state.drawMode || !state.drawStart || !state.drawPreview) {
          state.drawStart = null;
          state.drawPreview = null;
          return;
        }
        const label = ensureLabel();
        const width = image.clientWidth || 1;
        const height = image.clientHeight || 1;
        const p = state.drawPreview;
        if (p.w < 4 || p.h < 4) {
          state.drawStart = null;
          state.drawPreview = null;
          drawOverlay();
          return;
        }
        const x1 = Math.max(0, Math.min(1, p.x / width));
        const y1 = Math.max(0, Math.min(1, p.y / height));
        const x2 = Math.max(0, Math.min(1, (p.x + p.w) / width));
        const y2 = Math.max(0, Math.min(1, (p.y + p.h) / height));
        const evidenceId = nextEvidenceId(label);
        label.evidence_regions.push({
          evidence_id: evidenceId,
          page: Number(state.page),
          kind: "problem_description",
          bbox: [x1, y1, x2, y2],
        });
        state.selectedEvidenceIndex = label.evidence_regions.length - 1;
        state.drawStart = null;
        state.drawPreview = null;
        setDirty(true);
        renderEvidenceTable();
        drawOverlay();
      });
      window.addEventListener("resize", drawOverlay);
    }

    function wireEvents() {
      byId("prevBtn").addEventListener("click", () => gotoDocByIndex(state.index - 1));
      byId("nextBtn").addEventListener("click", () => gotoDocByIndex(state.index + 1));
      byId("saveBtn").addEventListener("click", () => saveCurrentLabel());
      byId("drawBtn").addEventListener("click", () => {
        state.drawMode = !state.drawMode;
        byId("drawBtn").style.borderColor = state.drawMode ? "#9a5a00" : "";
      });
      byId("pageSelect").addEventListener("change", () => {
        state.page = Number(byId("pageSelect").value || 1);
        state.selectedEvidenceIndex = -1;
        setImage();
      });
      byId("jumpBtn").addEventListener("click", () => {
        const value = Number(byId("jumpInput").value || -1);
        const idx = state.docs.findIndex((row) => Number(row.doc_id) === value);
        if (idx >= 0) gotoDocByIndex(idx);
      });
      byId("templateVersionInput").addEventListener("change", syncMetaFromInputs);
      byId("reviewStatusSelect").addEventListener("change", syncMetaFromInputs);
      byId("annotatorInput").addEventListener("change", syncMetaFromInputs);
      byId("proposalActionSelect").addEventListener("change", syncMetaFromInputs);

      byId("addEvidenceBtn").addEventListener("click", () => {
        const label = ensureLabel();
        if (!label) return;
        label.evidence_regions.push({
          evidence_id: nextEvidenceId(label),
          kind: "other",
          page: Number(state.page),
          bbox: [0.1, 0.1, 0.2, 0.2],
        });
        state.selectedEvidenceIndex = label.evidence_regions.length - 1;
        setDirty(true);
        renderEvidenceTable();
        drawOverlay();
      });

      byId("delEvidenceBtn").addEventListener("click", () => {
        const label = ensureLabel();
        if (!label) return;
        if (state.selectedEvidenceIndex < 0 || state.selectedEvidenceIndex >= label.evidence_regions.length) return;
        label.evidence_regions.splice(state.selectedEvidenceIndex, 1);
        state.selectedEvidenceIndex = -1;
        setDirty(true);
        renderEvidenceTable();
        drawOverlay();
      });

      byId("addProblemBtn").addEventListener("click", () => {
        const label = ensureLabel();
        if (!label) return;
        const uid = `p${(label.problems || []).length + 1}`;
        label.problems.push({
          problem_uid: uid,
          problem_number: "",
          description_text: "",
          description_evidence_ids: [],
          figure_evidence_ids: [],
        });
        setDirty(true);
        renderProblemTable();
      });

      byId("addRubricBtn").addEventListener("click", () => {
        const label = ensureLabel();
        if (!label) return;
        label.tier2.rubric_scores.push({
          problem_uid: "p1",
          criterion_id: "",
          max_points: 1,
          awarded_points: 1,
          error_tags: [],
          evidence_ids: [],
        });
        setDirty(true);
        renderRubricTable();
      });

      byId("addFeedbackBtn").addEventListener("click", () => {
        const label = ensureLabel();
        if (!label) return;
        label.tier2.feedback.push({
          problem_uid: "p1",
          comment: "",
          specificity: 0,
          actionability: 0,
          correctness: 1,
          evidence_ids: [],
        });
        setDirty(true);
        renderFeedbackTable();
      });

      byId("loadRubricBtn").addEventListener("click", applyRubricPreset);

      document.addEventListener("keydown", async (ev) => {
        if (ev.key === "ArrowRight") {
          gotoDocByIndex(state.index + 1);
        } else if (ev.key === "ArrowLeft") {
          gotoDocByIndex(state.index - 1);
        } else if ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "s") {
          ev.preventDefault();
          await saveCurrentLabel();
        } else if (ev.key.toLowerCase() === "b") {
          state.drawMode = !state.drawMode;
          byId("drawBtn").style.borderColor = state.drawMode ? "#9a5a00" : "";
        }
      });

      window.setInterval(async () => {
        if (!state.dirty) return;
        await saveCurrentLabel({silent: true});
      }, state.autosaveMs);
    }

    async function loadState() {
      const resp = await fetch("/state.json");
      if (!resp.ok) throw new Error(`Failed to load state (${resp.status})`);
      const payload = await resp.json();
      state.docs = Array.isArray(payload.docs) ? payload.docs : [];
      state.labels = payload.labels || {};
      state.proposals = payload.proposals || {};
      state.rubrics = payload.rubrics || {};
      state.autosaveMs = Number(payload.config?.autosave_ms || 15000);

      const savedDocId = Number(localStorage.getItem("taAnnotDocId") || -1);
      if (savedDocId >= 0) {
        const idx = state.docs.findIndex((row) => Number(row.doc_id) === savedDocId);
        if (idx >= 0) state.index = idx;
      }
      state.page = Number((currentDoc()?.pages?.[0]?.page) || 1);
      renderAll();
    }

    (async function boot() {
      wireEvents();
      setupCanvasInteractions();
      try {
        await loadState();
      } catch (err) {
        byId("status").innerHTML = `<span class="bad">${esc(err)}</span>`;
      }
    })();
  </script>
</body>
</html>
"""


class TAAnnotationRequestHandler(BaseHTTPRequestHandler):
    server: "TAAnnotationServer"

    def _send(
        self,
        *,
        status: int,
        body: bytes,
        content_type: str,
    ) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            html = build_html().encode("utf-8")
            self._send(status=HTTPStatus.OK, body=html, content_type="text/html; charset=utf-8")
            return
        if path == "/state.json":
            payload = self.server.build_state_payload()
            body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
            self._send(status=HTTPStatus.OK, body=body, content_type="application/json; charset=utf-8")
            return
        if path.startswith("/img/"):
            encoded_name = path[len("/img/") :]
            filename = unquote_path(encoded_name)
            resolved = self.server.resolve_image(filename)
            if resolved is None or not resolved.exists():
                body = b'{"message":"image not found"}'
                self._send(status=HTTPStatus.NOT_FOUND, body=body, content_type="application/json")
                return
            mime_type, _ = mimetypes.guess_type(str(resolved))
            content_type = mime_type or "application/octet-stream"
            with open(resolved, "rb") as f:
                data = f.read()
            self._send(status=HTTPStatus.OK, body=data, content_type=content_type)
            return

        self._send(
            status=HTTPStatus.NOT_FOUND,
            body=b'{"message":"not found"}',
            content_type="application/json; charset=utf-8",
        )

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/save":
            self._send(
                status=HTTPStatus.NOT_FOUND,
                body=b'{"message":"not found"}',
                content_type="application/json; charset=utf-8",
            )
            return

        content_len = _parse_int(self.headers.get("Content-Length"), 0)
        raw = self.rfile.read(max(0, content_len))
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            self._send(
                status=HTTPStatus.BAD_REQUEST,
                body=b'{"message":"invalid json"}',
                content_type="application/json; charset=utf-8",
            )
            return
        if not isinstance(payload, dict):
            self._send(
                status=HTTPStatus.BAD_REQUEST,
                body=b'{"message":"invalid payload"}',
                content_type="application/json; charset=utf-8",
            )
            return

        label = payload.get("label")
        if not isinstance(label, dict):
            self._send(
                status=HTTPStatus.BAD_REQUEST,
                body=b'{"message":"payload.label must be an object"}',
                content_type="application/json; charset=utf-8",
            )
            return

        ok, path, issues = save_label(
            labels_dir=self.server.labels_dir,
            payload=label,
            schema=self.server.schema,
            known_error_tags=self.server.known_error_tags,
        )
        if not ok:
            body = json.dumps({"message": "validation failed", "issues": issues}, ensure_ascii=True).encode("utf-8")
            self._send(
                status=HTTPStatus.BAD_REQUEST,
                body=body,
                content_type="application/json; charset=utf-8",
            )
            return

        doc_id = _parse_int(label.get("doc_id"), -1)
        if doc_id >= 0:
            self.server.labels_map[doc_id] = label
        body = json.dumps(
            {
                "ok": True,
                "path": path,
                "label": label,
            },
            ensure_ascii=True,
        ).encode("utf-8")
        self._send(status=HTTPStatus.OK, body=body, content_type="application/json; charset=utf-8")


class TAAnnotationServer(ThreadingHTTPServer):
    def __init__(
        self,
        server_address: tuple[str, int],
        handler_cls: type[BaseHTTPRequestHandler],
        *,
        images_dir: Path,
        docs_map: dict[int, dict[str, Any]],
        labels_dir: Path,
        labels_map: dict[int, dict[str, Any]],
        proposals_map: dict[int, dict[str, Any]],
        rubrics_map: dict[str, list[dict[str, Any]]],
        schema: dict[str, Any],
        known_error_tags: set[str],
        autosave_ms: int,
    ) -> None:
        super().__init__(server_address, handler_cls)
        self.images_dir = images_dir
        self.docs_map = docs_map
        self.labels_dir = labels_dir
        self.labels_map = labels_map
        self.proposals_map = proposals_map
        self.rubrics_map = rubrics_map
        self.schema = schema
        self.known_error_tags = known_error_tags
        self.autosave_ms = autosave_ms

    def resolve_image(self, filename: str) -> Optional[Path]:
        raw = str(filename).strip()
        if not raw:
            return None
        path_obj = Path(raw)
        if not path_obj.is_absolute():
            path_obj = (self.images_dir / path_obj).resolve(strict=False)
        return path_obj

    def build_state_payload(self) -> dict[str, Any]:
        docs_rows: list[dict[str, Any]] = []
        labels_rows: dict[str, dict[str, Any]] = {}
        proposals_rows: dict[str, dict[str, Any]] = {}
        for doc_id in sorted(self.docs_map):
            doc = self.docs_map[doc_id]
            docs_rows.append(doc)
            key = str(doc_id)

            proposal = self.proposals_map.get(doc_id)
            proposals_rows[key] = proposal or {}

            label = self.labels_map.get(doc_id)
            if label is None:
                label = _build_default_label(
                    doc_id=doc_id,
                    assignment_id=int(doc.get("assignment_id", 1)),
                    submission_id=str(doc.get("submission_id", "") or f"unknown-{doc_id}"),
                    proposal=proposal,
                )
            labels_rows[key] = label

        return {
            "docs": docs_rows,
            "labels": labels_rows,
            "proposals": proposals_rows,
            "rubrics": self.rubrics_map,
            "config": {
                "autosave_ms": int(self.autosave_ms),
                "schema_version": TA_SCHEMA_VERSION,
            },
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local UI for TA benchmark annotation.")
    parser.add_argument("--images-dir", required=True, help="Directory containing document images")
    parser.add_argument("--doc-info", required=True, help="Path to doc_info.csv")
    parser.add_argument("--labels-dir", required=True, help="Directory for label files")
    parser.add_argument("--test-ids", help="Optional path to test_ids.csv for assignment/submission metadata")
    parser.add_argument("--proposals", help="Optional proposal JSON path from ta-generate-proposals")
    parser.add_argument("--rubrics-dir", help="Optional rubric directory with <assignment_id>.yaml files")
    parser.add_argument("--schema", default=DEFAULT_TA_SCHEMA_PATH, help="TA schema path")
    parser.add_argument("--taxonomy", default=DEFAULT_TA_TAXONOMY_PATH, help="Error taxonomy path")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=8776, help="Server port")
    parser.add_argument("--autosave-ms", type=int, default=15000, help="Autosave interval in ms")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    console.print(f"[yellow]{PII_WARNING_BANNER}[/yellow]")

    images_dir = Path(args.images_dir).expanduser().resolve(strict=False)
    doc_info_csv = Path(args.doc_info).expanduser().resolve(strict=False)
    labels_dir = Path(args.labels_dir).expanduser().resolve(strict=False)
    labels_dir.mkdir(parents=True, exist_ok=True)

    test_ids_csv = Path(args.test_ids).expanduser().resolve(strict=False) if args.test_ids else None
    rubrics_dir = Path(args.rubrics_dir).expanduser().resolve(strict=False) if args.rubrics_dir else None

    docs_map = _load_doc_index(doc_info_csv=doc_info_csv, test_ids_csv=test_ids_csv, images_dir=images_dir)
    labels_map = _load_labels_map(labels_dir)
    proposals_map = _load_proposals_map(args.proposals)
    rubrics_map = _load_rubrics(rubrics_dir)
    schema = load_ta_schema(args.schema)
    known_error_tags = load_error_taxonomy(args.taxonomy)

    summary = Table(title="TA Annotation UI")
    summary.add_column("Field", style="cyan")
    summary.add_column("Value")
    summary.add_row("doc_info.csv", str(doc_info_csv))
    summary.add_row("images_dir", str(images_dir))
    summary.add_row("labels_dir", str(labels_dir))
    summary.add_row("docs", str(len(docs_map)))
    summary.add_row("existing labels", str(len(labels_map)))
    summary.add_row("proposals loaded", str(len(proposals_map)))
    summary.add_row("rubrics loaded", str(len(rubrics_map)))
    summary.add_row("autosave ms", str(int(args.autosave_ms)))
    console.print(summary)

    server = TAAnnotationServer(
        (args.host, int(args.port)),
        TAAnnotationRequestHandler,
        images_dir=images_dir,
        docs_map=docs_map,
        labels_dir=labels_dir,
        labels_map=labels_map,
        proposals_map=proposals_map,
        rubrics_map=rubrics_map,
        schema=schema,
        known_error_tags=known_error_tags,
        autosave_ms=int(args.autosave_ms),
    )

    url = f"http://{args.host}:{args.port}/"
    console.print(f"[green]Serving TA annotation UI at {url}[/green]")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("[yellow]Stopping server...[/yellow]")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
