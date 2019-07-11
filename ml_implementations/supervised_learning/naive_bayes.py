import numpy as np


class GaussianNaiveBayesClassifier:
    """A Guassian Naive Bayes Classifer.
    Classification is done using the Bayesian rule: P(Y|X) = P(X|Y) * P(Y)/P(X)
    The _naive_ in the name indicates we assume independence between features.

    Parameters:
    -----------
    None.
    """

    def __init__(self):
        self.y = None
        self.classes = None
        self.features_mean_var = []

    def fit(self, X, y):
        self.y = y
        self.classes = np.unique(y)
        for i, c in enumerate(self.classes):
            X_of_class = X[np.where(y == c)]
            self.features_mean_var.append([])
            for feature in X_of_class.T:
                feature_mean_var = {"mean": feature.mean(), "var": feature.var()}
                self.features_mean_var[i].append(feature_mean_var)

    @staticmethod
    def _calc_likelihood(mean, var, x, eps=1e-4):  # Likelihood: P(X|Y)
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var + eps)
        exponent = np.exp(-((x - mean) ** 2.0) / (2.0 * var + eps))
        likelihood = coeff * exponent
        return likelihood

    def _calc_prior(self, c):  # Prior: P(Y)
        freq = np.mean(self.y == c)
        return freq

    def _classify(self, sample):  # Posterior: P(Y|X)
        posteriors = []
        for i, c in enumerate(self.classes):
            posterior = self._calc_prior(c)
            for feature_value, params in zip(sample, self.features_mean_var[i]):
                likelihood = self._calc_likelihood(params["mean"], params["var"], feature_value)
                posterior *= likelihood
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]

    def predict(self, X):
        y_pred = [self._classify(sample) for sample in X]
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = (y_pred == y).mean()
        return accuracy
