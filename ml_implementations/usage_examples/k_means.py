import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from ml_implementations.unsupervised_learning.k_means import KMeans


def main():
    X, y = datasets.make_blobs()
    clf = KMeans(k=3)
    y_pred = clf.predict(X)
    plt.scatter(X[:, 0], X[:, 1], c=y_pred)
    plt.title('KMeans (k=3) Predicted Clusters')
    plt.show()


if __name__ == '__main__':
    main()
