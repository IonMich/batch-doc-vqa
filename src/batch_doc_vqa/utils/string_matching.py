import argparse
import json

import pandas as pd
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

try:
    from ..core import get_imagepaths_from_doc_info
except ImportError:  # pragma: no cover - fallback for direct script execution
    from batch_doc_vqa.core import get_imagepaths_from_doc_info

D_CUTOFF = 3


def levenshteinDistance(s1, s2):
    """Compute the Levenshtein distance between two strings.

    https://stackoverflow.com/a/32558749/10119867
    """
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            # following https://stackoverflow.com/a/31599276/10119867
            if c1.casefold() == c2.casefold():
                distances_.append(distances[i1])
            else:
                distances_.append(
                    1 + min((distances[i1], distances[i1 + 1], distances_[-1]))
                )
        distances = distances_
    return distances[-1]


def _load_results(results_filename: str) -> Dict[str, Any]:
    with open(results_filename, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Results JSON must be an object keyed by image path")
    return payload


def _resolve_doc_filename(raw_filename: Any, *, doc_info_filename: str, images_dir: Optional[str]) -> str:
    filename = str(raw_filename or "").strip()
    path = Path(filename)
    if path.is_absolute():
        return str(path)

    if images_dir:
        return str(Path(images_dir) / path)

    return str(Path(doc_info_filename).parent / path)


def _load_doc_info_df(
    doc_info_filename: str,
    *,
    pages: Optional[Sequence[int]],
    images_dir: Optional[str],
) -> pd.DataFrame:
    df = pd.read_csv(doc_info_filename)
    if pages is not None:
        df = df[df["page"].isin(list(pages))]
    df = df.copy()
    df["filename"] = df["filename"].apply(
        lambda item: _resolve_doc_filename(item, doc_info_filename=doc_info_filename, images_dir=images_dir)
    )
    return df


def _lookup_result_entry(
    *,
    results: Dict[str, Any],
    basename_map: Dict[str, Optional[Any]],
    filename: str,
) -> Optional[Any]:
    candidates = [
        filename,
        str(Path(filename)),
        Path(filename).as_posix(),
    ]

    path_obj = Path(filename)
    if path_obj.exists():
        candidates.append(str(path_obj.resolve()))

    for candidate in candidates:
        if candidate in results:
            return results[candidate]

    by_basename = basename_map.get(path_obj.name)
    if by_basename is not None:
        return by_basename
    return None


def _build_basename_index(results: Dict[str, Any]) -> Dict[str, Optional[Any]]:
    basename_map: Dict[str, Optional[Any]] = {}
    for key, value in results.items():
        base = Path(str(key)).name
        if base in basename_map:
            basename_map[base] = None
            continue
        basename_map[base] = value
    return basename_map


def get_llm_ids_and_fullnames(
    results_filename: str,
    doc_info_filename: str = "imgs/q11/doc_info.csv",
    *,
    pages: Optional[Sequence[int]] = (1, 3),
    images_dir: Optional[str] = None,
):
    results = _load_results(results_filename)
    filenames = get_imagepaths_from_doc_info(
        doc_info_filename,
        images_dir=images_dir,
        pages=pages,
    )
    basename_map = _build_basename_index(results)
    llm_ids = {}
    llm_fullnames = {}
    
    for filename in filenames:
        result_entry = _lookup_result_entry(
            results=results,
            basename_map=basename_map,
            filename=filename,
        )

        if isinstance(result_entry, list) and result_entry:
            first_entry = result_entry[0]
        else:
            first_entry = result_entry

        if isinstance(first_entry, dict):
            # Get the first valid result (skip failed entries)
            llm_ids[filename] = first_entry.get("university_id", "")
            llm_fullnames[filename] = first_entry.get("student_full_name", "")
        else:
            # Handle missing or empty results
            print(f"Warning: No results found for {filename}")
            llm_ids[filename] = ""
            llm_fullnames[filename] = ""
    df_ids = pd.DataFrame(llm_ids.items(), columns=["filename", "llm_id"])
    df_fullnames = pd.DataFrame(
        llm_fullnames.items(), columns=["filename", "llm_fullname"]
    )
    df = pd.merge(df_ids, df_fullnames, on="filename")

    return df


def get_llm_distances(
    df_llm,
    doc_info_filename,
    ids_filename,
    *,
    pages: Optional[Sequence[int]] = (1, 3),
    images_dir: Optional[str] = None,
):
    df_filenames = _load_doc_info_df(
        doc_info_filename,
        pages=pages,
        images_dir=images_dir,
    )
    df_llm = df_llm.merge(df_filenames, on="filename")

    df_test = pd.read_csv(ids_filename)
    #TODO: doc column should be removed from ids_filename file
    df_test = df_test.drop(columns=["doc"])
    df_test = df_llm.merge(df_test, how="cross")
    # compare IDs
    df_test["llm_id"] = df_test["llm_id"].astype(str)
    df_test["student_id"] = df_test["student_id"].astype(str)
    df_test["id_distance"] = df_test.apply(
        lambda x: levenshteinDistance(x["llm_id"], x["student_id"]), axis=1
    )
    # compare last names
    df_test["lastname_distance"] = df_test.apply(
        lambda x: levenshteinDistance(
            x["llm_fullname"].split()[-1] if x["llm_fullname"] else "",
            x["student_full_name"].split()[-1] if x["student_full_name"] else "",
        ),
        axis=1,
    )
    n_id_correct = df_test["id_distance"].eq(0).sum()
    n_lastname_correct = df_test["lastname_distance"].eq(0).sum()
    print(f"Number of perfect ID matches: {n_id_correct}/{len(df_test)}")
    print(f"Number of perfect last name matches: {n_lastname_correct}/{len(df_test)}")
    return df_test


def get_matches(df_test, id_d_cutoff=D_CUTOFF):
    df = df_test[
        (df_test["id_distance"] <= id_d_cutoff) | (df_test["lastname_distance"] == 0)
    ].copy()
    
    df_matching = df.groupby(["doc", "student_id"]).agg(
        {
            "filename": "first",
            "student_id": "first",
            "student_full_name": "first",
            "id_distance": "min",
            "lastname_distance": "min",
        }
    )
    df_matching["found"] = df_matching.apply(
        lambda x: (x["id_distance"] == 0)
        or ((x["id_distance"] <= id_d_cutoff) and (x["lastname_distance"] == 0)),
        axis=1,
    )
    return df_matching


def parse_llm_pipe(
    results_filename,
    doc_info_filename,
    ids_filename,
    store_filename,
    id_d_cutoff=D_CUTOFF,
    pages: Optional[Sequence[int]] = (1, 3),
    images_dir: Optional[str] = None,
):
    df_llm = get_llm_ids_and_fullnames(
        results_filename,
        doc_info_filename,
        pages=pages,
        images_dir=images_dir,
    )
    df_test = get_llm_distances(
        df_llm,
        doc_info_filename,
        ids_filename,
        pages=pages,
        images_dir=images_dir,
    )
    df_matching = get_matches(df_test, id_d_cutoff)
    df_matching.to_csv(store_filename, index=False)
    return df_matching


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_fname",
        type=str,
        default="tests/output/qwen2-VL-2B-results.json",
        help="Filename of the results JSON file",
    )
    parser.add_argument(
        "--doc_info_fname",
        type=str,
        default="imgs/q11/doc_info.csv",
        help="Filename of the document info CSV file",
    )
    parser.add_argument(
        "--ids_fname",
        type=str,
        default="tests/data/test_ids.csv",
        help="Filename of the test IDs CSV file",
    )
    parser.add_argument(
        "--store_fname",
        type=str,
        default="tests/output/qwen2-VL-2B-matching_results.csv",
        help="Filename to store the matching results",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default=None,
        help="Optional images directory for resolving relative filenames in doc_info.csv",
    )
    parser.add_argument(
        "--pages",
        type=str,
        default="1,3",
        help="Comma-separated page numbers to include (default: 1,3)",
    )

    args = parser.parse_args()
    pages = tuple(int(p.strip()) for p in args.pages.split(",") if p.strip())
    df_matching = parse_llm_pipe(
        args.results_fname,
        args.doc_info_fname,
        args.ids_fname,
        args.store_fname,
        pages=pages,
        images_dir=args.images_dir,
    )
    print(df_matching)
