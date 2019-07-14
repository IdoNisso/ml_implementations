import numpy as np


class PCA:
    """Principal Component Analysis, a dimensionality reduction method.
    Returns a lower number of features for X by getting rid of correlation and maximizing
    variance along each new feature axis.

    Parameters:
    -----------
    None.
    """
    def __init__(self):
        pass

    def fitpredict(self, X, num_components):
        covariance_matrix = self._calc_covariance_matrix(X)
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
        idx = eigenvalues.argsort()[::-1]
        eigenvectors = np.atleast_1d(eigenvectors[:, idx])[:, :num_components]
        X_ = X.dot(eigenvectors)
        return X_

    @staticmethod
    def _calc_covariance_matrix(X):
        n_samples = np.shape(X)[0]
        covariance_matrix = (1 / (n_samples - 1)) * (X - X.mean(axis=0)).T.dot(X - X.mean(axis=0))
        return np.array(covariance_matrix, dtype=float)
