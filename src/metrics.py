import numpy as np
import pandas as pd
from numpy import logical_and, sum as t_sum


def naive_accuracy(
        predicted_labels: list([list]), true_labels: pd.DataFrame, num_labels: int
) -> float:
    total_matches = 0
    for i in range(1, true_labels.shape[0] - 1):
        truth = true_labels.loc[i, :].values.tolist()[1:]
        predicted = list(predicted_labels[i])
        if truth == predicted:
            total_matches += 1

    return total_matches / (true_labels.shape[0] - 1)


def label_frequency(true_labels: pd.DataFrame, num_labels: int) -> np.ndarray:
    counts = []
    for i in range(num_labels):
        counts.append(true_labels.iloc[:, i + 1].value_counts()[1])

    num_args = true_labels.shape[0] - 1

    return np.array([i / num_args for i in counts])


# Precision = True Positives / All Positives Predictions
def precision(predicted_labels: list, true_labels: pd.DataFrame, which_label: int = 1):
    pred_which = np.array([pred == which_label for pred in predicted_labels])
    true_which = np.array([lab == which_label for lab in true_labels])
    denominator = t_sum(pred_which)

    if denominator:
        return t_sum(logical_and(pred_which, true_which)) / denominator
    else:
        return 0.
