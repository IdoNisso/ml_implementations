import numpy as np


class KNNClassifier:
    """K Nearest Neighbors classifier.
    Given samples and labels, will classify an unknown labeled sample using its k nearest neighbors.

    Parameters:
    -----------
    k_neighbors: int
        The number of closest neighbors to use when determining sample class.
    """
    def __init__(self, k_neighbors):
        self.k_neighbors = k_neighbors
        self.training_set = None
        self.training_label = None

    def fit(self, X, y):
        self.training_set = X
        self.training_label = y

    def predict(self, X):
        predictions = np.empty(X.shape[0], dtype=int)
        for s in range(X.shape[0]):
            predictions[s] = self._predict_one_sample(X[s])
        return predictions

    def score(self, X, y):
        predictions = self.predict(X)
        mean_accuracy = sum(predictions == y) / float(len(y))
        return mean_accuracy

    def _predict_one_sample(self, x):
        distances = np.sqrt(np.sum((self.training_set - x) ** 2, axis=1))  # Euclidean distance
        index_array = np.argsort(distances)
        knn_labels = self.training_label[index_array[0:self.k_neighbors]]
        unique, counts = np.unique(knn_labels, return_counts=True)
        prediction = unique[np.argmax(counts)]
        return prediction
