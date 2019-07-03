import numpy as np


class OrdinaryLeastSquares:
    """Ordinary Least Squares Linear Regression.

    Parameters:
    -----------
    None.
    """
    def __init__(self, pad=True):
        self.pad = pad
        self.weights = None

    def fit(self, X, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        self._fit(X, y)

    @staticmethod
    def _pad(X):
        return np.pad(X, ((0, 0), (1, 0)), 'constant', constant_values=1)

    def _fit(self, X, y):
        if self.pad:
            X = self._pad(X)
        X_pseudo_inverse = np.linalg.pinv(X)  # Moore-Penrose pseudo-inverse for more efficiency
        self.weights = np.dot(X_pseudo_inverse, y)

    def predict(self, X):
        if self.weights is None:
            raise ValueError('no weights, run fit(X, y) to fit model before prediction')
        return self._predict(X)

    def _predict(self, X):
        if self.pad:
            y_pred = np.dot(self._pad(X), self.weights)
        else:
            y_pred = np.dot(X, self.weights)
        return y_pred

    def score(self, X, y):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        y_pred = self.predict(X)
        mse = np.square(y_pred - y).mean()
        return mse


class OrdinaryLeastSquaresGD(OrdinaryLeastSquares):
    """Ordinary Least Squares Linear Regression with Gradient Descent.

    Parameters:
    -----------
    learning_rate: float
        The rate at which to update learned weights. Also known as alpha.
    n_iter: int
        Number of iterations to run in order to learn weights.
    early_stop: boolean
        Allows to stop the learning process if loss has plateaued.
    """
    def __init__(self, learning_rate=0.05, n_iter=1000, early_stop=True):
        super().__init__()
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.early_stop = early_stop
        self.loss = []

    def _fit(self, X, y):
        if self.pad:
            X = self._pad(X)
        self.weights = np.zeros((X.shape[1], 1))
        for iter in range(self.n_iter):
            # print('iter %d/%d' % (iter, self.n_iter))
            self._step(X, y)
            curr_loss = self.score(X[:, 1:], y)
            self.loss.append(curr_loss)
            if self.early_stop and (len(self.loss) >= 3) and (self.loss[-2] < self.loss[-1]):
                # print('early stop!')
                break

    def _step(self, X, y):
        n = X.shape[0]
        y_est = np.dot(X, self.weights)
        error = y_est - y
        grad = np.dot(X.T, error) / n
        self.weights = self.weights - (self.learning_rate * grad)
