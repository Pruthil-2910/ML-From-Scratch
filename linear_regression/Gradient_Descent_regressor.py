"""
Simple Linear Regression implemented from scratch using Python.
Author: Pruthil Prajapati

"""
import numpy as np
class GDRegressor:
  def __init__(self, Learning_Rate, Epochs):
    self.Learning_Rate = Learning_Rate
    self.Epochs = Epochs
    self.coef = None
    self.intercept = None

  def fit(self, X_train, Y_train):
    self.intercept = 0
    self.coef = np.ones(X_train.shape[1])
    for i in range(self.Epochs):
      Y_hat = np.dot(X_train, self.coef) + self.intercept
      intercept_der = -2 * np.mean(Y_train - Y_hat)
      self.intercept -= self.Learning_Rate * intercept_der
      coef_der = -2 * np.dot((Y_train - Y_hat), X_train) / X_train.shape[0]
      self.coef -= self.Learning_Rate * coef_der

  def predict(self, X_test):
    if self.coef is not None:
        return np.dot(X_test, self.coef) + self.intercept

  def r2_score(self, Y_true, Y_pred):
    ss_total = np.sum((Y_true - Y_true.mean()) ** 2)
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    return 1 - (ss_res / ss_total)


model = GDRegressor(0.01,1000)
"""
model.fit(X_train , Y_train)
Y_pred = model.predict(X_test)
print(model.r2_score(Y_true,Y_pred))

"""
