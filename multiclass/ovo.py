import random
import numpy as np
import pandas as pd
from collections import Counter
import pSMO
from tqdm import tqdm

class OvOClf():
    def __init__(self, X, y, base, args={}, ifTqdm=False):
        random.seed(42)
        self.samples = X
        self.labels, self.trans = pd.factorize(np.array(y))
        self.cnt = len(self.trans)
        self.clfs = []
        if ifTqdm:
            print("Start Training ...")
            pbar = tqdm(total=(self.cnt - 1) * self.cnt // 2)

        for i in range(self.cnt - 1):
            for j in range(i + 1, self.cnt):
                cur_samples_i = self.samples[self.labels == i]
                cur_samples_j = self.samples[self.labels == j]
                cur_samples = np.vstack((cur_samples_i, cur_samples_j))
                cur_labels = np.ones((1, len(cur_samples_i) + len(cur_samples_j)))
                cur_labels[0, len(cur_samples_i):] = -1
                cur_labels = np.array(cur_labels[0])
                arr = list(range(len(cur_samples)))
                random.shuffle(arr)
                arr = np.array(arr)
                cur_samples = cur_samples[arr]
                cur_labels = cur_labels[arr]
                clf = base()
                clf.fit(cur_samples, cur_labels, **args)
                self.clfs.append(clf)
                if ifTqdm:
                    pbar.update(1)
        
        if ifTqdm:
            pbar.close()


    def predict(self, X):
        # Set seed for reproducibility
        random.seed(42)
        y = []
        all_votes = np.zeros((len(X), self.cnt *(self.cnt - 1) // 2), dtype=int)
        vote_idx = 0

        for i in range(self.cnt - 1):
            for j in range(i + 1, self.cnt):
                results = self.clfs[vote_idx].predict(X)
                all_votes[:, vote_idx] = np.where(results == 1, i, j)
                vote_idx += 1

        for votes in all_votes:
            count = Counter(votes)
            max_count = max(count.values())
            modes = [k for k, v in count.items() if v == max_count]
            y.append(self.trans[modes[0]])

        return np.array(y)
