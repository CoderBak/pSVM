# In this file, we load all the datasets that are needed.
# All the labels of binary classification tasks are formatted into {-1, 1}

import numpy as np
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import scale

def load_cancer():
    # Breast Cancer Wisconsin (Diagnostic)
    breast_cancer = load_breast_cancer()
    X = breast_cancer.data
    y = breast_cancer.target
    trans = np.vectorize(lambda x: 1 if x == 1 else -1)
    print("### Loading {} Dataset. Size: {} x {} ###".format(
        "Breast Cancer",
        len(X), len(X[0])
    ))
    return X, trans(y)

def load_heart():
    # Statlog (Heart)
    statlog_heart = fetch_ucirepo(id=145)
    X = statlog_heart.data.features.values
    y = statlog_heart.data.targets.values.T[0]
    trans = np.vectorize(lambda x: 1 if x == 2 else -1)
    print("### Loading {} Dataset. Size: {} x {} ###".format(
        "Heart",
        len(X), len(X[0])
    ))
    return X, trans(y)

def load_ionosphere():
    # Ionosphere
    ionosphere = fetch_ucirepo(id=52)
    X = ionosphere.data.features.values
    y = ionosphere.data.targets.values.T[0]
    trans = np.vectorize(lambda x: 1 if x == 'g' else -1)
    print("### Loading {} Dataset. Size: {} x {} ###".format(
        "Ionosphere",
        len(X), len(X[0])
    ))
    return X, trans(y)

def load_wine():
    # Wine Quality
    wine_quality = fetch_ucirepo(id=186)
    X = wine_quality.data.features.values
    y = wine_quality.data.targets.values.T[0]
    trans = np.vectorize(lambda x: -1 if x <= 5 else 1)
    print("### Loading {} Dataset. Size: {} x {} ###".format(
        "Wine Quality",
        len(X), len(X[0])
    ))
    return X, trans(y)

def load_banknote():
    # Banknote Authentication
    banknote_authentication = fetch_ucirepo(id=267)
    X = banknote_authentication.data.features.values
    y = banknote_authentication.data.targets.values.T[0]
    trans = np.vectorize(lambda x: 1 if x == 1 else -1)
    print("### Loading {} Dataset. Size: {} x {} ###".format(
        "Banknote",
        len(X), len(X[0])
    ))
    return X, trans(y)

class Dataset:
    def __init__(self, name, need_scale=True):
        if name == "heart":
            self.samples, self.labels = load_heart()
        if name == "breast_cancer":
            self.samples, self.labels = load_cancer()
        if name == "ionosphere":
            self.samples, self.labels = load_ionosphere()
        if name == "wine":
            self.samples, self.labels = load_wine()
        if name == "banknote":
            self.samples, self.labels = load_banknote()

        if need_scale:
            self.samples = scale(self.samples)
