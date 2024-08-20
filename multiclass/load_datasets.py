# In this file, we load all the datasets that are needed.

import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import scale

def load_glass():
    # Glass
    glass_identification = fetch_ucirepo(id=42)
    X = glass_identification.data.features.values
    y = glass_identification.data.targets.values.T[0]
    print("### Loading {} Dataset with {} classes. Size: {} x {} ###".format(
        "Glass", len(pd.factorize(y)[1]),
        len(X), len(X[0])
    ))
    return X, y

def load_vehicle():
    # Vehicle
    statlog_vehicle_silhouettes = fetch_ucirepo(id=149)
    X = statlog_vehicle_silhouettes.data.features
    y = statlog_vehicle_silhouettes.data.targets
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()
    X = combined.iloc[:, :-1].values
    y = combined.iloc[:, -1].values
    print("### Loading {} Dataset with {} classes. Size: {} x {} ###".format(
        "Vehicle", len(pd.factorize(y)[1]),
        len(X), len(X[0])
    ))
    return X, y

def load_dermatology():
    dermatology = fetch_ucirepo(id=33)
    X = dermatology.data.features
    y = dermatology.data.targets
    combined = pd.concat([X, y], axis=1)
    combined = combined.dropna()
    X = combined.iloc[:, :-1].values
    y = combined.iloc[:, -1].values
    print("### Loading {} Dataset with {} classes. Size: {} x {} ###".format(
        "Dermatology", len(pd.factorize(y)[1]),
        len(X), len(X[0])
    ))
    return X, y

def load_USPS():
    data, target = fetch_openml(data_id=41082, return_X_y=True)
    X = data.values
    y = target.values
    print("### Loading {} Dataset with {} classes. Size: {} x {} ###".format(
        "USPS", len(pd.factorize(y)[1]),
        len(X), len(X[0])
    ))
    return X, y

class Dataset:
    def __init__(self, name, need_scale=True):
        if name == "glass":
            self.samples, self.labels = load_glass()
        if name == "vehicle":
            self.samples, self.labels = load_vehicle()
        if name == "dermatology":
            self.samples, self.labels = load_dermatology()
        if name == "USPS":
            self.samples, self.labels = load_USPS()

        if need_scale:
            self.samples = scale(self.samples)
