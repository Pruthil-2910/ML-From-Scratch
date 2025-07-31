"""
Simple Linear Regression implemented from scratch using Python.
Author: Pruthil Prajapati

"""
import numpy as np

class myLinearRegression:
    def __init__(self):
        self.m = None
        self.b = None

    def fit(self, Xtrain, Ytrain):
        den = 0
        num = 0
        for i in range(Xtrain.shape[0]):

            num += (Xtrain[i] - Xtrain.mean()) * (Ytrain[i] - Ytrain.mean())
            den += (Xtrain[i] - Xtrain.mean()) ** 2

        self.m = num / den
        self.b = Ytrain.mean() - (self.m * Xtrain.mean())

    def predict(self , Xtest):
        return self.m * Xtest + self.b
    
    def r2_score(self, Ytrue, Ypred):
        ss_total = np.sum((Ytrue - Ytrue.mean()) ** 2)
        ss_res = np.sum((Ytrue - Ypred) ** 2)
        return 1 - (ss_res / ss_total)

model = myLinearRegression()

"""

Now you can you use the model on the parameters X_Train and Y_Train as given below and predict the Y_Pred
model.fit(X_Train ,Y_Train)
Y_Pred = model.predict(X_Test)
score = model.r2_score(Y_train, Y_train_pred)

"""