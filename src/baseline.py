import random
import pandas as pd
import numpy as np
from util import load_training_data
from metrics import naive_accuracy, label_frequency

NUM_LABELS = 20


def predict_random(training_labels: pd.DataFrame, num_samples):
    predictions = []
    for i in range(NUM_LABELS):
        weights = [
            training_labels.iloc[:, i + 1].value_counts()[0],
            training_labels.iloc[:, i + 1].value_counts()[1],
        ]
        predictions.append(random.choices([0, 1], weights=weights, k=num_samples))

    return np.transpose(predictions)


print("Loading Data")
training_args, training_labels = load_training_data()
predictions = predict_random(training_labels, training_labels.shape[0] - 1)
# print(training_labels.shape)
# print(np.shape(predictions))
print("Load Complete")

np.set_printoptions(precision=2)
print("Frequency of Training Labels")
print(label_frequency(training_labels, NUM_LABELS))

print(
    f"Baseline Test: Random Guessing, Naive Accuracy (num correct / num args): {naive_accuracy(predictions, training_labels, NUM_LABELS)}"
)
