from sklearn import datasets
from sklearn.model_selection import train_test_split
from ml_implementations.supervised_learning.logisitic_regression import LogisticRegression


def main():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Training set sample size: ', X_train.shape[0], '\nTest set sample size: ', X_test.shape[0], '\n')

    gd = LogisticRegression(method='GD')
    gd.fit(X_train, y_train)
    y_pred_gd = gd.predict(X_test)
    print('GD Accuracy: ', (y_pred_gd == y_test).mean())

    sgd = LogisticRegression(method='SGD')
    sgd.fit(X_train, y_train)
    y_pred_sgd = sgd.predict(X_test)
    print('SGD Accuracy: ', (y_pred_sgd == y_test).mean())


if __name__ == '__main__':
    main()
