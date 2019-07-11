from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from ml_implementations.supervised_learning.linear_regression import OrdinaryLeastSquares, OrdinaryLeastSquaresGD


def main():
    X, y = datasets.load_boston(return_X_y=True)
    X = normalize(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print('\nOrdinary Least Squares:')
    model = OrdinaryLeastSquares()
    model.fit(X_train, y_train)
    train_mse = model.score(X_train, y_train)
    print('Training MSE: %f' % train_mse)
    test_mse = model.score(X_test, y_test)
    print('Test MSE: %f' % test_mse)

    print('\nOrdinary Least Squares Gradient Descent:')
    learning_rates = [0.1, 0.05, 0.01]
    for lr in learning_rates:
        model = OrdinaryLeastSquaresGD(learning_rate=lr)
        model.fit(X_train, y_train)
        train_mse = model.score(X_train, y_train)
        print('Training MSE (lr = %f): %f' % (lr, train_mse))
        test_mse = model.score(X_test, y_test)
        print('Test MSE (lr = %f): %f' % (lr, test_mse))


if __name__ == '__main__':
    main()
