import numpy as np


class KMeans:
    """An unsupervised clustering algorithm that iteratively assigns samples to the closest centroid and
    moves the centroid to the center of the new clusters.

    Parameters:
    -----------
    k : int
        Number of clusters to separate the data into.
    max_iterations : int
        Max number of iterations to run. The fitting process will stop before reaching max_iterations if it
        reaches convergence.
    """
    def __init__(self, k=2, max_iterations=1000):
        self.k = k
        self.max_iterations = max_iterations

    def fit(self):
        raise ValueError("KMeans doesn't need to fit. Run predict!")

    def predict(self, X):
        centroids = self._init_centroids(X)
        for iteration in range(self.max_iterations):
            prev_centroids = centroids
            clusters = self._create_clusters(X, centroids)
            centroids = self._move_centroids(X, clusters)
            if not (centroids - prev_centroids).any():
                print('No change between iterations, clusters converged at iteration :', iteration)
                break
        return self._get_labels(X, clusters)

    def _init_centroids(self, X):
        n_samples, n_features = np.shape(X)
        centroids = np.zeros((self.k, n_features))
        for k_i in range(self.k):
            centroid = X[np.random.choice(range(n_samples))]
            centroids[k_i] = centroid
        return centroids

    def _create_clusters(self, X, centroids):
        clusters = [[] for _ in range(self.k)]
        for sample_i, sample in enumerate(X):
            centroid_i = self._find_closest_centroid(sample, centroids)
            clusters[centroid_i].append(sample_i)
        return clusters

    def _move_centroids(self, X, clusters):
        n_features = np.shape(X)[1]
        centroids = np.zeros((self.k, n_features))
        for i, cluster in enumerate(clusters):
            centroids[i] = np.mean(X[cluster], axis=0)
        return centroids

    def _find_closest_centroid(self, sample, centroids):
        closest_i = 0
        closest_dist = float('inf')
        for i, centroid in enumerate(centroids):
            distance = self._euclidean_distance(sample, centroid)
            if distance < closest_dist:
                closest_i = i
                closest_dist = distance
        return closest_i

    @staticmethod
    def _get_labels(X, clusters):
        y_pred = np.zeros(np.shape(X)[0])
        for cluster_i, cluster in enumerate(clusters):
            for sample_i in cluster:
                y_pred[sample_i] = cluster_i
        return y_pred

    @staticmethod
    def _euclidean_distance(x_1, x_2):
        distance = 0
        for idx in range(len(x_1)):
            distance += (x_1[idx] - x_2[idx]) ** 2
        return np.sqrt(distance)
