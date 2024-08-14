# A modified version of svmpy package.

import numpy as np
import cvxpy as cp

MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5

class SVMTrainer(object):
    def __init__(self, kernel, c, p):
        self._kernel = kernel
        self._c = c
        self._p = p
        self._gamma = p / (p - 1) if self._p != 1 else 2.0
        self._theta = (c ** (1 - self._gamma)) * (p ** (-self._gamma)) * (p - 1)

    def train(self, X, y):
        """Given the training features X with labels y, returns a SVM
        predictor representing the trained SVM.
        """
        lagrange_multipliers = self._compute_multipliers(X, y)
        return self._construct_predictor(X, y, lagrange_multipliers)

    def _gram_matrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))
        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i, j] = self._kernel(x_i, x_j)
        return K

    def _construct_predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = \
            lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]
        support_vector_labels = y[support_vector_indices]

        # http://www.cs.cmu.edu/~guestrin/Class/10701-S07/Slides/kernels.pdf
        # bias = y_k - \sum z_i y_i  K(x_k, x_i)
        # Thus we can just predict an example with bias of zero, and
        # compute error.
        bias = np.mean(
            [y_k - SVMPredictor(
                kernel=self._kernel,
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            kernel=self._kernel,
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def _compute_multipliers(self, X, y):
        n_samples, n_features = X.shape

        K = self._gram_matrix(X)
        P = np.outer(y, y) * K
        G = np.diag(np.ones(n_samples) * -1)
        h = np.ravel(np.zeros(n_samples))
        A = np.asmatrix(y).reshape(1, -1)
        b = 0.0

        x = cp.Variable(n_samples)

        if self._p != 1:
            prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P)
                                        - cp.sum(x)
                                        + self._theta * cp.sum(cp.power(x, self._gamma))),
                            [G @ x <= h, A @ x == b])
        else:
            prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, P)
                                        - cp.sum(x)),
                            [G @ x <= h, A @ x == b])
        prob.solve(solver=cp.GUROBI)

        return np.ravel(x.value)


class SVMPredictor(object):
    def __init__(self,
                 kernel,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._kernel = kernel
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels
        assert len(support_vectors) == len(support_vector_labels)
        assert len(weights) == len(support_vector_labels)

    def predict(self, x):
        """
        Computes the SVM prediction on the given features x.
        """
        result = self._bias
        for z_i, x_i, y_i in zip(self._weights,
                                 self._support_vectors,
                                 self._support_vector_labels):
            result += z_i * y_i * self._kernel(x_i, x)
        return np.sign(result).item()
