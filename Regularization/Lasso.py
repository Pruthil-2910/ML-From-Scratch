
"""
Simple Linear Regression implemented from scratch by Gradient Descent Approach using Python.
Author: Pruthil Prajapati

"""
import numpy as np

class LassoRegressionGD:
    def __init__(self, lambda_=1.0, learning_rate=0.01, epochs=1000):
        self.lambda_ = lambda_
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef = None
        self.intercept = None
        self.loss_history = []

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.coef = np.zeros(n_features)
        self.intercept = 0

        for epoch in range(self.epochs):
            y_hat = np.dot(X_train, self.coef) + self.intercept
            loss = np.mean((y_train - y_hat) ** 2) + self.lambda_ * np.sum(np.abs(self.coef))
            self.loss_history.append(loss)

            intercept_gradient = -2 * np.mean(y_train - y_hat)
            coef_gradient = -2 * np.dot(X_train.T, (y_train - y_hat)) / n_samples

            l1_penalty_gradient = self.lambda_ * np.sign(self.coef)
            coef_gradient += l1_penalty_gradient

            self.intercept -= self.learning_rate * intercept_gradient
            self.coef -= self.learning_rate * coef_gradient

    def predict(self, X_test):
        if self.coef is None:
            raise Exception("Model has not been fitted yet.")
        return np.dot(X_test, self.coef) + self.intercept

    def r2_score(self, y_true, y_pred):
        ss_total = np.sum((y_true - y_true.mean()) ** 2)
        ss_res = np.sum((y_true - y_pred) ** 2)
        return 1 - (ss_res / ss_total)
    
my_lasso = LassoRegressionGD(lambda_ =1.0) 

# my_lasso.fit(X_train, y_train)
# my_y_pred = my_lasso.predict(X_test)
# my_r2 = my_lasso.r2_score(y_test, my_y_pred)