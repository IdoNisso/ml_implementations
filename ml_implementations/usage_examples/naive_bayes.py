from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from ml_implementations.supervised_learning.naive_bayes import GaussianNaiveBayesClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def main():
    X, y = datasets.load_digits(return_X_y=True)
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = GaussianNaiveBayesClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = clf.score(X_test, y_test)
    print('Gaussian Naive Bayes accuracy: ', accuracy)
    plt.matshow(confusion_matrix(y_pred, y_test), cmap='gray')
    plt.show()


if __name__ == '__main__':
    main()
