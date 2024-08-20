import cupy as cp
import numpy as np
import random
from math import sqrt

# pSVM implementation for binary classification
class SVM():
    def __init__(self, p=1.5, max_iter=10000, eps=1e-6):
        self.p = p
        self.max_iter = max_iter
        self.eps = eps

    def fit(self, X, y, C=1):
        n = len(X)
        self.X = X
        self.y = y
        X = cp.asarray(X)  # X = np.asarray(X)   (without cupy)
        y = np.asarray(y)
        self.b = np.mean(y)

        count = 0
        self.a = np.zeros((n))
        E = -self.y
        gamma = self.p / (self.p - 1)
        theta = (C ** (1 - gamma)) * (self.p ** (- gamma)) * (self.p - 1)
        g_t = gamma * theta

        random.seed(42)

        self.K = (X @ X.T).get()  # self.K = X @ X.T   (without cupy)

        while True:
            a_prev = np.copy(self.a)
            for j in range(n):
                count += 1
                i = self.get_random(0, n - 1, j)

                # The update process in paper
                a_old_i = self.a[i]
                a_old_j = self.a[j]
                c = a_old_i * y[i] + a_old_j * y[j]
                eta_ij = self.K[i][i] + self.K[j][j] - 2 * self.K[i][j]

                if eta_ij == 0:
                    continue

                absc = abs(c)
                Q_ij = eta_ij * a_old_j + y[j] * (E[i] - E[j]) - g_t * (absc ** (gamma - 1))
                
                if y[i] == y[j]:
                    condition1 = (- 2 * g_t * (absc ** (gamma - 1)) <= Q_ij)
                    condition2 = (Q_ij <= eta_ij * absc)
                    if condition1 and condition2:
                        if self.p == 1.5:
                            a_j = (Q_ij + (absc ** 2) * 6 * theta) / (eta_ij + 6 * absc * theta)
                        else:
                            a_j = (Q_ij + 4 * theta * absc) / (eta_ij + 4 * theta)
                    elif not condition1:
                        a_j = 0
                    else:
                        a_j = absc
                else:
                    u = max(0, - c * y[i])
                    if Q_ij <= eta_ij * u:
                        a_j = u
                    else:
                        if c * y[i] >= 0:
                            if self.p == 1.5:
                                a_j = (sqrt((eta_ij + 6 * absc * theta) ** 2 + 24 * Q_ij * theta) - 6 * absc * theta - eta_ij) / (12 * theta)
                            else:
                                a_j = Q_ij / (eta_ij + 4 * theta)
                        else:
                            if self.p == 1.5:
                                a_j = (sqrt((eta_ij - 6 * absc * theta) ** 2 + 24 * Q_ij * theta) + 6 * absc * theta - eta_ij) / (12 * theta)
                            else:
                                a_j = (Q_ij + 4 * absc * theta) / (eta_ij + 4 * theta)

                a_i = a_old_i + y[i] * y[j] * (a_old_j - a_j)

                E = E + (self.K[:, i] * (a_i - self.a[i]) * self.y[i] + self.K[:, j] * (a_j - self.a[j]) * self.y[j])

                self.a[i] = a_i
                self.a[j] = a_j

                # E = self.K @ (self.a * self.y) - self.y

                sv = self.a > 0
                self.b = - np.mean(E[sv]) if np.any(sv) else np.mean(self.y)
                if count >= self.max_iter:
                    break

            diff = np.linalg.norm(self.a - a_prev)
            if diff < self.eps or count >= self.max_iter:
                break

    def predict(self, target):
        results = (self.a * self.y) @ self.X @ target.T + self.b
        return np.where(results >= 0, 1, -1)

    def get_random(self, a, b, z):
        i = z
        for _ in range(1000):
            i = random.randint(a, b)
            if i != z:
                return i
        return random.randint(a, b)
