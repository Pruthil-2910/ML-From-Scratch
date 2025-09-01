"""
Ridge (Linear)Regression using Gradient Descent implemented from scratch using Python.
Author: Pruthil Prajapati

"""
import numpy as np

class RidgeGD:
  def __init__(self, epochs, learning_rate, alpha):
    self.learning_rate = learning_rate
    self.epochs = epochs
    self.alpha = alpha
    self.coef = None
    self.intercept = None
    self.loss_history = []

  def fit(self,X_train, y_train):
    self.coef = np.ones(X_train.shape[1])
    self.intercept = 0
    X_train_b = np.insert(X_train, 0, 1, axis=1)
    theta = np.insert(self.coef, 0, self.intercept)


    for i in range(self.epochs):
      theta_der = np.dot(X_train_b.T, X_train_b).dot(theta) - np.dot(X_train_b.T, y_train) + self.alpha * theta
      loss = np.mean((y_train - np.dot(X_train_b, theta)) ** 2)
      theta = theta - self.learning_rate * theta_der

    self.intercept = theta[0]
    self.coef = theta[1:]

  def predict(self, X_test):
    if self.coef is not None:
        return np.dot(X_test, self.coef) + self.intercept

  def r2_score(self, Y_true, Y_pred):
    ss_total = np.sum((Y_true - Y_true.mean()) ** 2)
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    return 1 - (ss_res / ss_total)