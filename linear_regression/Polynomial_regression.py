"""
Polynomial Regression implemented from scratch using Python.
Author: Pruthil Prajapati

"""


import numpy as np

class PolynomialRegression:
    def __init__(self, degree=1):
        self.degree = degree
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        X_poly = self._create_polynomial_features(X)
        betas = np.linalg.inv(X_poly.T @ X_poly) @ X_poly.T @ y
        self.intercept_ = betas[0]
        self.coef_ = betas[1:]

    def predict(self, X):
        if self.coef_ is not None:
          X_poly = self._create_polynomial_features(X)
          return X_poly @ self.coef_ + self.intercept_

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - y_true.mean()) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_total)

    def _create_polynomial_features(self, X):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        X_poly = X
        for i in range(1, self.degree + 1):
            X_poly = np.hstack((X_poly, X**i))
        return X_poly
    
degree = 3
model = PolynomialRegression(degree)

"""

Now you can you use the model on the parameters X_Train and Y_Train as given below and predict the Y_Pred
model.fit(X_Train ,Y_Train)
Y_Pred = model.predict(X_Test)
score = model.r2_score(Y_train, Y_train_pred)

"""