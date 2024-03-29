from sklearn import datasets
from sklearn.model_selection import train_test_split
from ml_implementations.supervised_learning.k_nearest_neighbors import KNNClassifier


def main():
    data = datasets.load_iris()
    X = data.data
    y = data.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k = 3
    clf = KNNClassifier(k)
    clf.fit(X_train, y_train)
    # print(clf.predict(X_test))

    accuracy = clf.score(X_test, y_test)
    print('Accuracy (k=%i): %f' % (k, accuracy))


if __name__ == '__main__':
    main()
