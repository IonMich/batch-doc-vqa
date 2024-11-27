import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def get_prob_calibration(pred_filename, ids_filename):
    df_probs = pd.read_csv(pred_filename)
    df_ids = pd.read_csv(ids_filename)

    df_ids["student_id"] = df_ids["student_id"].apply(str)
    df_ids["student_id"] = df_ids["student_id"].apply(lambda x: x.zfill(8))
    df_ids = df_ids["student_id"].apply(lambda x: pd.Series(list(x)))
    df_ids = df_ids.stack().reset_index(level=1, drop=True).to_frame("digit")
    df_ids = df_ids.reset_index()
    df_ids = df_ids.rename(columns={"index": "doc_idx"})
    df_ids["digit"] = df_ids["digit"].astype(int)

    df_ids = pd.get_dummies(df_ids, columns=["digit"], prefix="digit")

    new_df = pd.merge(df_probs, df_ids, left_index=True, right_index=True)
    new_df = pd.DataFrame(
        {
            "probability": new_df.iloc[:, 0:10].values.flatten(),
            "is_correct": new_df.iloc[:, 11:].values.flatten(),
        }
    )

    bins = np.linspace(0, 1, 8)
    new_df["probability_bin"] = pd.cut(new_df["probability"], bins)
    new_df["probability_bin"] = new_df["probability_bin"].apply(
        lambda x: (x.left + x.right) / 2
    )
    new_df = new_df.groupby("probability_bin", observed=True)["is_correct"].mean()
    
    return new_df

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
    hist = plt.hist(df_probs, bins=20)
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.title("Predicted digit probabilities")
    plt.savefig(f"tests/output/prob_hist_{label}.png")
    return hist

def get_max_prob_histogram(pred_filename, label):
    """
    Get histograms of max predicted probabilities
    """
    df_probs = pd.read_csv(pred_filename)
    df_probs = df_probs.max(axis=1)
    print(df_probs)
    hist = plt.hist(df_probs, bins=20)
    plt.xlabel("Probability")
    plt.ylabel("Frequency")
    plt.title("Max predicted digit probabilities")
    plt.savefig(f"tests/output/max_prob_hist_{label}.png")
    return hist