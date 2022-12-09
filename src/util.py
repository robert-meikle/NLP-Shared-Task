import numpy as np
import pandas as pd

args_type_dict = {"Argument ID": str, "Conclusion": str, "Stance": str, "Premise": str}
labels_type_dict = {
    "Argument ID": str,
    "Self-direction: thought": int,
    "Self-direction: action": int,
    "Stimulation": int,
    "Hedonism": int,
    "Achievement": int,
    "Power: dominance": int,
    "Power: resources": int,
    "Face": int,
    "Security: personal": int,
    "Security: societal": int,
    "Tradition": int,
    "Conformity: rules": int,
    "Conformity: interpersonal": int,
    "Humility": int,
    "Benevolence: caring": int,
    "Benevolence: dependability": int,
    "Universalism: concern": int,
    "Universalism: nature": int,
    "Universalism: tolerance": int,
    "Universalism: objectivity": int,
}


def load_training_data(args: str, labels: str):
    args_path = args
    labels_path = labels

    arguments = []
    with open(args_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        #print(lines[0])
        # skip reading header
        for line in lines[1:]:
            temp = []
            stance = "in favor of"
            if "against" in line:
                stance = "against"
                temp = line.split("against")
            elif "in favor of" in line:
                temp = line.split("in favor of")
            elif "in favour of" in line:
                temp = line.split("in favour of")
            else:
                print(f"unrecognized stance for line '{line}', skipping.")
                continue
            arguments.append(
                [
                    temp[0].split()[0],
                    " ".join(temp[0].split()[1:]),
                    stance,
                    temp[1].strip(),
                ]
            )

    arguments = pd.DataFrame(
        arguments, columns=["Argument ID", "Conclusion", "Stance", "Premise"]
    )
    arguments = arguments.astype(args_type_dict)
    print(arguments["Stance"].head(10))

    labels = []
    with open(labels_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        # skipping header again, adding column names manually
        for line in lines[1:]:
            labels.append(line.split())

    labels = pd.DataFrame(
        labels,
        columns=[
            "Argument ID",
            "Self-direction: thought",
            "Self-direction: action",
            "Stimulation",
            "Hedonism",
            "Achievement",
            "Power: dominance",
            "Power: resources",
            "Face",
            "Security: personal",
            "Security: societal",
            "Tradition",
            "Conformity: rules",
            "Conformity: interpersonal",
            "Humility",
            "Benevolence: caring",
            "Benevolence: dependability",
            "Universalism: concern",
            "Universalism: nature",
            "Universalism: tolerance",
            "Universalism: objectivity",
        ],
    ).astype(labels_type_dict)
    #print(labels.head(10))

    return arguments, labels
