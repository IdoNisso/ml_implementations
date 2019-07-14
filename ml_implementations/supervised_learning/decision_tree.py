import numpy as np


class DecisionTree:
    """A Decision tree that may be used for regression and classification problems.
    Builds itself recursively during 'fit'.

    Parameters:
    -----------
    min_samples_split: int
        The minimum number of samples needed to make a split when building a tree.
    min_impurity: float
        The minimum impurity required to split the tree further.
    max_depth: int
        The maximum depth of a tree.
    tree_type : string
        Either 'classification' or 'regression'.
    """
    def __init__(self, min_samples_split=2, min_impurity=1e-7, max_depth=float("inf"), tree_type='classification'):
        self.root = None
        self.min_samples_split = min_samples_split
        self.min_impurity = min_impurity
        self.max_depth = max_depth

        if tree_type not in ['classification', 'regression']:
            raise ValueError('invalid tree type, must be classification or regression!')
        self.tree_type = tree_type
        if tree_type == 'classification':
            self._impurity_calculation = self._calc_information_gain
            self._leaf_value_calculation = self._majority_vote
        else:  # 'regression'
            self._impurity_calculation = self._calc_variance_reduction
            self._leaf_value_calculation = self._mean_of_y

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def predict(self, X):
        y_pred = [self._predict_value(sample) for sample in X]
        return y_pred

    def score(self, X, y):
        if self.tree_type == 'classification':
            y_pred = self.predict(X)
            accuracy = (y_pred == y).mean()
            return accuracy
        else:  # 'regression'
            y_pred = self.predict(X)
            mse = np.square(y_pred - y).mean()
            return mse

    def _build_tree(self, X, y, curr_depth=0):
        largest_impurity = 0
        best_criteria = None
        best_sets = None

        if len(np.shape(y)) == 1:
            y = np.expand_dims(y, axis=1)

        Xy = np.concatenate((X, y), axis=1)

        n_samples, n_features = np.shape(X)

        if n_samples >= self.min_samples_split and curr_depth <= self.max_depth:
            for feature_i in range(n_features):
                feature_values = np.expand_dims(X[:, feature_i], axis=1)
                unique_values = np.unique(feature_values)

                for threshold in unique_values:
                    Xy1, Xy2 = self._divide_on_feature(Xy, feature_i, threshold)

                    if len(Xy1) > 0 and len(Xy2) > 0:
                        y1 = Xy1[:, n_features:]
                        y2 = Xy2[:, n_features:]

                        impurity = self._impurity_calculation(y, y1, y2)

                        if impurity > largest_impurity:
                            largest_impurity = impurity
                            best_criteria = {"feature_idx": feature_i, "threshold": threshold}
                            best_sets = {
                                "left_X": Xy1[:, :n_features],
                                "left_y": Xy1[:, n_features:],
                                "right_X": Xy2[:, :n_features],
                                "right_y": Xy2[:, n_features:]
                            }

        if largest_impurity > self.min_impurity:
            left_branch = self._build_tree(best_sets["left_X"], best_sets["left_y"], curr_depth + 1)
            right_branch = self._build_tree(best_sets["right_X"], best_sets["right_y"], curr_depth + 1)
            return DecisionNode(feature_idx=best_criteria["feature_idx"], threshold=best_criteria["threshold"],
                                left_branch=left_branch, right_branch=right_branch)

        leaf_value = self._leaf_value_calculation(y)
        return DecisionNode(value=leaf_value)

    def _predict_value(self, x, tree=None):
        if tree is None:
            tree = self.root
        if tree.value is not None:
            return tree.value
        feature_value = x[tree.feature_idx]
        if feature_value > tree.threshold:
            branch = tree.left_branch
        else:
            branch = tree.right_branch
        return self._predict_value(x, tree=branch)

    def _calc_variance_reduction(self, y, y1, y2):
        var_tot = self._calc_variance(y)
        var_1 = self._calc_variance(y1)
        var_2 = self._calc_variance(y2)
        frac_1 = len(y1) / len(y)
        frac_2 = len(y2) / len(y)
        variance_reduction = var_tot - (frac_1 * var_1 + frac_2 * var_2)
        return sum(variance_reduction)

    def _calc_information_gain(self, y, y1, y2):
        p = len(y1) / len(y)
        entropy = self._calc_entropy(y)
        info_gain = entropy - p * self._calc_entropy(y1) - (1 - p) * self._calc_entropy(y2)
        return info_gain

    @staticmethod
    def _divide_on_feature(X, feature_idx, threshold):
        split_func = lambda sample: sample[feature_idx] >= threshold
        X_1 = np.array([sample for sample in X if split_func(sample)])
        X_2 = np.array([sample for sample in X if not split_func(sample)])
        return np.array([X_1, X_2])

    @staticmethod
    def _mean_of_y(y):
        value = np.mean(y, axis=0)
        return value if len(value) > 1 else value[0]

    @staticmethod
    def _majority_vote(y):
        most_common = None
        max_count = 0
        for label in np.unique(y):
            count = len(y[y == label])
            if count > max_count:
                most_common = label
                max_count = count
        return most_common

    @staticmethod
    def _calc_variance(X):
        mean = np.ones(np.shape(X)) * X.mean(0)
        n_samples = np.shape(X)[0]
        variance = (1 / n_samples) * np.diag((X - mean).T.dot(X - mean))
        return variance

    @staticmethod
    def _calc_entropy(y):
        log2 = lambda x: np.log(x) / np.log(2)
        unique_labels = np.unique(y)
        entropy = 0
        for label in unique_labels:
            count = len(y[y == label])
            p = count / len(y)
            entropy += -p * log2(p)
        return entropy


class DecisionNode:
    """Represents a decision node or value leaf in a decision tree.

    Parameters:
    -----------
    feature_idx : int
        The index of the feature which node uses.
    threshold : float
        Threshold value to decide between left and right using features[feature_idx].
    value : int / float
        The prediction value (class int or float).
    left_branch : DecisionNode
        Next node for sample deemed 'True' - above threshold value.
    right_branch : DecisionNode
        Next node for sample deemed 'False' - below/equal threshold value.
    """
    def __init__(self, feature_idx=None, threshold=None, value=None, left_branch=None, right_branch=None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.value = value
        self.left_branch = left_branch
        self.right_branch = right_branch
