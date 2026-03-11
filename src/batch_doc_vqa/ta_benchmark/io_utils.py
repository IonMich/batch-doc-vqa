"""I/O helpers for TA benchmark tooling."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Optional


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(Path(path).expanduser().resolve(strict=False), "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: str | Path) -> Any:
    with open(Path(path).expanduser().resolve(strict=False), "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path).expanduser().resolve(strict=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)
    return out


def load_doc_info(path: str | Path) -> dict[int, list[dict[str, Any]]]:
    csv_path = Path(path).expanduser().resolve(strict=False)
    docs: dict[int, list[dict[str, Any]]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue
            try:
                doc = int(str(row.get("doc", "")).strip())
                page = int(str(row.get("page", "")).strip())
            except ValueError:
                continue
            filename = str(row.get("filename", "")).strip()
            if not filename:
                continue
            docs.setdefault(doc, []).append(
                {
                    "doc": doc,
                    "page": page,
                    "filename": filename,
                }
            )

    for items in docs.values():
        items.sort(key=lambda item: int(item["page"]))
    return docs


def load_test_ids(path: str | Path) -> dict[int, dict[str, str]]:
    csv_path = Path(path).expanduser().resolve(strict=False)
    gt: dict[int, dict[str, str]] = {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row is None:
                continue
            raw_doc = str(row.get("doc", "")).strip()
            if not raw_doc:
                continue
            try:
                doc = int(raw_doc)
            except ValueError:
                continue
            cleaned = {str(key): str(value or "").strip() for key, value in row.items() if key is not None}
            gt[doc] = cleaned
    return gt


def build_result_basename_index(results: dict[str, Any]) -> dict[str, Optional[Any]]:
    index: dict[str, Optional[Any]] = {}
    for key, value in results.items():
        basename = Path(str(key)).name
        if basename in index:
            index[basename] = None
            continue
        index[basename] = value
    return index


def lookup_result_entry(
    *,
    results: dict[str, Any],
    basename_index: dict[str, Optional[Any]],
    filename: str,
    images_dir: Optional[str] = None,
) -> Optional[Any]:
    path_obj = Path(filename)
    resolved = (Path(images_dir) / path_obj if images_dir else path_obj).resolve(strict=False)
    candidates = [
        str(path_obj),
        path_obj.as_posix(),
        str(resolved),
        path_obj.name,
    ]
    for candidate in candidates:
        if candidate in results:
            return results[candidate]
    return basename_index.get(path_obj.name)
