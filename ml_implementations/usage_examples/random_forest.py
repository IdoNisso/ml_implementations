from sklearn import datasets
from sklearn.model_selection import train_test_split
from ml_implementations.supervised_learning.random_forest import RandomForest


def main():
    iris = datasets.load_iris()
    X = iris.data[:, :2]
    y = (iris.target != 0) * 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print('Training set sample size: ', X_train.shape[0], '\nTest set sample size: ', X_test.shape[0], '\n')

    clf = RandomForest()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)
    print('Random Forest test set accuracy on Iris: ', accuracy)


if __name__ == '__main__':
    main()
