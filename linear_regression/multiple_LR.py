import numpy as np

class MultipleLR:
  def __init__(self):
    self.coef_ = None
    self.intercept_ = None

  def fit(self , X_train , Y_train):
    X_train = np.insert(X_train , 0 , 1 , axis= 1)

    betas = np.linalg.inv(np.dot(X_train.T , X_train)).dot(X_train.T).dot(Y_train)
    self.intercept_ = betas[0]
    self.coef_ = betas[1:]

  def predict(self , X_test):
    if self.coef_ is not None:
        return np.dot(X_test , self.coef_) + self.intercept_

  def r2_score(self, Y_true, Y_pred):
    ss_total = np.sum((Y_true - Y_true.mean()) ** 2)
    ss_res = np.sum((Y_true - Y_pred) ** 2)
    return 1 - (ss_res / ss_total)

model = MultipleLR()

"""
model.fit(X_train , Y_train)
Y_pred = model.predict(X_test)
print(model.r2_score(Y_true,Y_pred))

"""
