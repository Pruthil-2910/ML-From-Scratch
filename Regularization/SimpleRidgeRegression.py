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


my_ridge = SimpleRidgeRegression(alpha=1.0) # You can adjust alpha
# my_ridge.fit(X_train, y_train)
# my_y_pred = my_ridge.predict(X_test)
# my_r2 = my_ridge.r2_score(y_test, my_y_pred)

