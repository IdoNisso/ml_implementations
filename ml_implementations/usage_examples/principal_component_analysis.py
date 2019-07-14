import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from ml_implementations.unsupervised_learning.principal_component_analysis import PCA


def main():
    X, y = datasets.load_digits(return_X_y=True)
    X_ = PCA().fitpredict(X, 2)
    x1 = X_[:, 0]
    x2 = X_[:, 1]

    class_dist = []
    for i, l in enumerate(np.unique(y)):
        _x1 = x1[y == l]
        _x2 = x2[y == l]
        _y = y[y == l]
        class_dist.append(plt.scatter(_x1, _x2))
    plt.show()


if __name__ == '__main__':
    main()