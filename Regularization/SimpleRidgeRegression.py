"""
Simple Ridge (Linear)Regression implemented from scratch using Python.
Author: Pruthil Prajapati

"""
import numpy as np
class SimpleRidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.b = None
        self.m = None
        
    def fit(self, Xtrain, Ytrain ):
        den, num = 0, 0
        for i in range(Xtrain.shape[0]):
            num += (Ytrain[i] - Ytrain.mean())*(Xtrain[i] - Xtrain.mean())
            den += (Xtrain[i] - Xtrain.mean()) ** 2
            
        self.m = num / (den + self.alpha)
        self.b = Ytrain.mean() - (self.m * Xtrain.mean())
        
    def predict(self , Xtest):
        return self.m * Xtest + self.b
    
    def r2_score(self, Ytrue, Ypred):
        ss_total = np.sum((Ytrue - Ytrue.mean()) ** 2)
        ss_res = np.sum((Ytrue - Ypred) ** 2)
        return 1 - (ss_res / ss_total)


model = SimpleRidgeRegression()

"""

Now you can you use the model on the parameters X_Train and Y_Train as given below and predict the Y_Pred
model.fit(X_Train ,Y_Train)
Y_Pred = model.predict(X_Test)
score = model.r2_score(Y_train, Y_train_pred)

"""
