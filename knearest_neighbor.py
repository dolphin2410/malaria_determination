from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from dataset_manager import read_dataset


def build_knn_model():
    raw_dataset, raw_answers = read_dataset()

    k_candidates = list(range(1,121))
    
    scaler = StandardScaler()
    X = X.reshape(-1, 1)
    print("a")
    X = scaler.fit_transform(X)

    scores = []

    for k in k_candidates:
        knn = KNeighborsClassifier(n_neighbors=k)
        score = cross_val_score(knn, raw_dataset, raw_answers, cv=30)
        scores.append(np.mean(score))

    plt.plot(k_candidates, scores)
    plt.xlabel("K Value")
    plt.ylabel("Accuracy")
    plt.show()