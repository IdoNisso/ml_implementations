from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from ml_implementations.supervised_learning.decision_tree import DecisionTree


def main():
    X, y = datasets.load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    clf = DecisionTree(tree_type='classification')
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('Classification DT accuracy on Iris: ', accuracy)

    X, y = datasets.load_boston(return_X_y=True)
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = DecisionTree(tree_type='regression')
    model.fit(X_train, y_train)
    mse = model.score(X_test, y_test)
    print('Regression DT MSE on Boston: ', mse)


if __name__ == '__main__':
    main()
