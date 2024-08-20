import numpy as np
from sklearn.model_selection import train_test_split

# Self-defined utils
import pSMO
from ovo import OvOClf
from load_datasets import Dataset

def eval(dataset, p, C=1, max_iter=5000, tqdm=False, eps=1e-4):
    print(f"Dataset: {dataset}, p: {p}")
    dataset = Dataset(dataset)

    train_sample, test_sample, train_label, test_label = train_test_split(
        dataset.samples, dataset.labels, test_size=0.2, random_state=42
    )

    clf = OvOClf(train_sample, train_label, lambda: pSMO.SVM(p=p, max_iter=max_iter, eps=eps), {'C': C}, tqdm)
    print(np.mean(clf.predict(test_sample) == test_label))


eval("glass", 2, 4.5, 5000)
eval("glass", 1.5, 1, 5000)
eval("dermatology", 2, 3.5, 5000)
eval("dermatology", 1.5, 2, 5000)
eval("vehicle", 2, 4, 10000)
eval("vehicle", 1.5, 2.25, 10000)
eval("USPS", 2, 0.25, 300000, True)
eval("USPS", 1.5, 0.25, 100000, True)
