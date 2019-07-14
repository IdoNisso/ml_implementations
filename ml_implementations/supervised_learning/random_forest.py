import numpy as np
from ml_implementations.supervised_learning.decision_tree import DecisionTree


class RandomForest:
    """A Random Forest classifier.

    Parameters:
    -----------
    num_estimators : int
        The number of decision trees to build in the forest.
    max_features : int
        The max number of features the trees may use.
    min_samples_split : int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth : int
        The maximum depth of a tree.
    """
    def __init__(self, num_estimators=100, max_features=None, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf")):
        self.num_estimators = num_estimators
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth
        self.trees = []
        for _ in range(self.num_estimators):
            self.trees.append(DecisionTree(self.min_samples_split, self.min_impurity, self.max_depth, 'classification'))

    def fit(self, X, y):
        num_features = np.shape(X)[1]
        if not self.max_features:
            self.max_features = int(num_features / 2)
        subsets = self._get_random_subsets(X, y, self.num_estimators)
        for clf_idx in range(self.num_estimators):
            # random subset of X and y
            X_subset, y_subset = subsets[clf_idx]
            # random subset of features
            feature_idxs = np.random.choice(range(num_features), size=self.max_features, replace=True)
            self.trees[clf_idx].feature_idxs = feature_idxs
            X_subset = X_subset[:, feature_idxs]
            self.trees[clf_idx].fit(X_subset, y_subset)

    def predict(self, X):
        y_preds = np.empty((X.shape[0], len(self.trees)))
        for i, tree in enumerate(self.trees):
            y_preds[:, i] = tree.predict(X[:, tree.feature_idxs])
        y_pred = []
        for sample_pred in y_preds:
            y_pred.append(np.bincount(sample_pred.astype(int)).argmax())
        return y_pred

    def score(self, X, y):
        y_pred = self.predict(X)
        accuracy = (y_pred == y).mean()
        return accuracy

    @staticmethod
    def _get_random_subsets(X, y, num_subsets):
        n_samples = np.shape(X)[0]
        X_y = np.concatenate((X, y.reshape((1, len(y))).T), axis=1)
        np.random.shuffle(X_y)
        subsets = []
        for _ in range(num_subsets):
            idx = np.random.choice(
                range(n_samples),
                size=np.shape(range(n_samples)),
                replace=True)
            X = X_y[idx][:, :-1]
            y = X_y[idx][:, -1]
            subsets.append([X, y])
        return subsets
