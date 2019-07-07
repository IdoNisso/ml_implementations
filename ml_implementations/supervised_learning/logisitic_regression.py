import numpy as np


class LogisticRegression:
    """Logistic Regression binary classifier.

    Parameters:
    -----------
    learning_rate : float
        The rate at which to update learned weights. Also known as alpha.
    n_iter: int
        Number of iterations to run in order to learn weights.
    method : str
        'GD' (Gradient Descent) or 'SGD' (Stochastic Gradient Descent)
    normalize : boolean
        Normalize the X before fitting with mean and std.
    add_intercept : boolean
        Add intercept column to X before fitting.
    """
    def __init__(self, learning_rate=0.01, n_iter=10000, method='GD', normalize=True, add_intercept=True):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.method = method
        self.normalize = normalize
        self.add_intercept = add_intercept
        self.weights = None
        self.X_mean = None
        self.X_std = None

    @staticmethod
    def _add_intercept(X):
        intercept = np.ones((X.shape[0], 1))
        return np.concatenate((intercept, X), axis=1)

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        if self.normalize:
            X = self._normalize(X)
        if self.add_intercept:
            X = self._add_intercept(X)
        self.weights = np.zeros(X.shape[1])  # initialize weights
        for iteration in range(self.n_iter):
            self._step(X, y)

    def _normalize(self, X):
        if not self.X_mean or not self.X_std:
            self.X_mean = X.mean()
            self.X_std = X.std()
        return (X - self.X_mean) / self.X_std

    def _step(self, X, y):
        n = X.shape[0]
        if self.method == 'GD':
            z = np.dot(X, self.weights)
            h = self._sigmoid(z)
            grad = np.dot(X.T, (h - y)) / n
            self.weights -= self.learning_rate * grad
        elif self.method == 'SGD':
            sample_indexes = np.random.permutation(n)
            for sample_idx in sample_indexes:
                z = np.dot(X[sample_idx, :], self.weights)
                h = self._sigmoid(z)
                grad = X[sample_idx, :].T * (h - y)[sample_idx]
                self.weights -= self.learning_rate * grad
        else:
            raise ValueError('Only GD or SGD supported')

    def predict(self, X, threshold=0.5):
        return self.predict_proba(X) >= threshold

    def predict_proba(self, X):
        if self.weights is not None:
            if self.normalize:
                X = self._normalize(X)
            return self._sigmoid(np.dot(self._add_intercept(X), self.weights))
        else:
            raise ValueError('Model needs to be fitted before prediction.')
