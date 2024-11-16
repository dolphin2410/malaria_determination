import csv
import numpy as np


def read_dataset():
    raw_dataset = []
    raw_answers = []

    with open("wbcprbc_dataset.CSV", "r") as dataset_file:
        dataset_reader = csv.reader(dataset_file)

        for rbc, wbc, rbcprbc, label in dataset_reader:
            rbc, wbc, rbcprbc, label = float(rbc), float(wbc), float(rbcprbc), int(label)

            if wbc == 0:
                continue

            label = label - 1 if label > 0 else label

            raw_dataset.append(rbcprbc)
            raw_answers.append(label)

    raw_dataset, raw_answers = np.array(raw_dataset), np.array(raw_answers)

    return raw_dataset, raw_answers