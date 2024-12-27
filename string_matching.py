import argparse

import pandas as pd
from outlines_quiz import json_load_results, get_imagepaths

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


def get_llm_ids_and_fullnames(results_filename):
    pages = [1, 3]
    results = json_load_results(results_filename)
    folder = "imgs/q11"
    pattern = r"doc-\d+-page-[" + "|".join([str(p) for p in pages]) + r"]-[A-Z0-9]+.png"
    print(f"Pattern: {pattern}")
    filenames = get_imagepaths(folder, pattern)
    llm_ids = {
        filename: results[filename][0]["university_id"] for filename in filenames
    }
    llm_fullnames = {
        filename: results[filename][0]["student_full_name"] for filename in filenames
    }
    df_ids = pd.DataFrame(llm_ids.items(), columns=["filename", "llm_id"])
    df_fullnames = pd.DataFrame(
        llm_fullnames.items(), columns=["filename", "llm_fullname"]
    )
    df = pd.merge(df_ids, df_fullnames, on="filename")

    return df


def get_llm_distances(df_llm, doc_info_filename, ids_filename):
    pages = [1, 3]
    df_filenames = pd.read_csv(doc_info_filename)
    # TODO: fix filepaths
    df_filenames["filename"] = "imgs/q11/" + df_filenames["filename"]
    query_str = " or ".join([f"page == {i}" for i in pages])
    df_filenames = df_filenames.query(query_str)
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
):
    df_llm = get_llm_ids_and_fullnames(results_filename)
    df_test = get_llm_distances(df_llm, doc_info_filename, ids_filename)
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

    args = parser.parse_args()
    df_matching = parse_llm_pipe(
        args.results_fname,
        args.doc_info_fname,
        args.ids_fname,
        args.store_fname,
    )
    print(df_matching)
