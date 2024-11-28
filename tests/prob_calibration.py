import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

UNI_ID_LENGTH = 8
DETECTION_PROB = 10 ** (3 - UNI_ID_LENGTH)


def get_prob_calibration(pred_filename, ids_filename):
    df_probs = pd.read_csv(pred_filename)
    df_ids = pd.read_csv(ids_filename)

    df_ids["student_id"] = df_ids["student_id"].apply(str)
    df_ids["student_id"] = df_ids["student_id"].apply(lambda x: x.zfill(UNI_ID_LENGTH))
    df_ids = df_ids["student_id"].apply(lambda x: pd.Series(list(x)))
    df_ids = df_ids.stack().reset_index(level=1, drop=True).to_frame("digit")
    df_ids = df_ids.reset_index()
    df_ids = df_ids.rename(columns={"index": "doc_idx"})
    df_ids["digit"] = df_ids["digit"].astype(int)

    df_ids = pd.get_dummies(df_ids, columns=["digit"], prefix="digit")

    new_df = pd.merge(df_probs, df_ids, left_index=True, right_index=True)
    is_correct = pd.DataFrame(
        {
            "probability": new_df.iloc[:, 0:10].values.flatten(),
            "is_correct": new_df.iloc[:, 11:].values.flatten(),
        }
    )

    bins = np.linspace(0, 1, 8)
    is_correct["probability_bin"] = pd.cut(is_correct["probability"], bins)
    is_correct["probability_bin"] = is_correct["probability_bin"].apply(
        lambda x: (x.left + x.right) / 2
    )
    new_df = is_correct.groupby("probability_bin", observed=True)["is_correct"].mean()

    return new_df, is_correct


def plot_calibration_curves(list_dfs, list_labels, save_path):
    for i, new_df in enumerate(list_dfs):
        plt.plot(new_df.index, new_df.values, marker="o", label=list_labels[i])
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("Probability")
    plt.ylabel("Accuracy")

    plt.title("Calibration plot")
    plt.legend()
    plt.grid()
    plt.savefig(save_path)


def get_histogram(pred_filename, label):
    """
    Get histograms of predicted probabilities
    """
    df_probs = pd.read_csv(pred_filename)
    df_probs = df_probs.values.flatten()
    hist = plt.hist(df_probs, bins=20, label=label)
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.title("Predicted digit probabilities")
    plt.legend()
    plt.savefig(f"tests/output/prob_hist_{label}.png")
    return hist


def get_max_prob_histogram(pred_filename, label):
    """
    Get histograms of max predicted probabilities
    """
    df_probs = pd.read_csv(pred_filename)
    df_probs = df_probs.max(axis=1)
    print(df_probs)
    hist = plt.hist(df_probs, bins=20, label=label)
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.title("Max predicted digit probabilities")
    plt.legend()
    plt.savefig(f"tests/output/max_prob_hist_{label}.png")
    return hist


def get_percent_correct_IDs(is_correct_df):
    """Return the percentage of correct predictions for the IDs in the dataset."""
    # filter out the incorrect predictions
    is_correct_df = is_correct_df[is_correct_df["is_correct"]]
    # group rows to get the total probability for each image
    is_correct_df = (
        is_correct_df["probability"]
        .groupby(is_correct_df.index // (UNI_ID_LENGTH * 10))
        .prod()
    )
    is_correct_df = (
        is_correct_df.groupby(is_correct_df > DETECTION_PROB)
        .count()
        .apply(lambda x: x / is_correct_df.count())
        .apply(lambda x: f"{x:.2%}")
    )
    return is_correct_df[True]


def create_summary_tables(pred_filenames, ids_filename, labels):
    """Generate a markdown summary table that contains:
    - top-1 accuracy percentage per digit
    - top-2 accuracy percentage per digit
    - top-3 accuracy percentage per digit
    - top-1 accuracy percentage per group of UNI_ID_LENDTH-digits ID

    Uses dataframe.to_markdown() to generate markdown table.
    """
    df_ids = pd.read_csv(ids_filename)
    df_ids["student_id"] = df_ids["student_id"].apply(str)
    df_ids["student_id"] = df_ids["student_id"].apply(lambda x: x.zfill(UNI_ID_LENGTH))
    df_ids = df_ids["student_id"].apply(lambda x: pd.Series(list(x)))
    df_ids = df_ids.stack().reset_index(level=1, drop=True).to_frame("digit")
    df_ids = df_ids.reset_index()
    df_ids = df_ids.rename(columns={"index": "doc_idx"})
    df_ids["digit"] = df_ids["digit"].astype(int)
    top_k_accs = []
    for pred_filename, label in zip(pred_filenames, labels):
        df_probs = pd.read_csv(pred_filename)
        df_probs = pd.merge(df_probs, df_ids, left_index=True, right_index=True)
        # print(df_probs)
        # keep only top-1 prediction probabilities from columns "0" to "9"
        # get the digit with the highest probability
        for k in [1, 2, 3]:
            df_probs["top{}_pred".format(k)] = df_probs[
                [str(i) for i in range(10)]
            ].apply(lambda x: x.nlargest(k).index.tolist(), axis=1)
            df_probs["top{}_correct".format(k)] = df_probs.apply(
                lambda x: str(x["digit"]) in x["top{}_pred".format(k)], axis=1
            )
        # get the overall top-k prediction accuracy with .mean()
        top_k_accs.append(
            df_probs[["top1_correct", "top2_correct", "top3_correct"]]
            .mean()
            .apply(lambda x: f"{x:.2%}")
            .rename(label)
        )
        _, is_correct = get_prob_calibration(pred_filename, ids_filename)
        percent_correct_IDs = get_percent_correct_IDs(is_correct)
        top_k_accs[-1]["IDs detect"] = percent_correct_IDs
    top_k_acc = pd.concat(top_k_accs, axis=1)
    print(top_k_acc.to_markdown())
    # save the table to a markdown file
    top_k_acc.to_markdown("tests/output/top_k_acc.md")
