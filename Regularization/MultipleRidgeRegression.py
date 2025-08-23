"""
Multiple Ridge (Linear)Regression implemented from scratch using Python.
Author: Pruthil Prajapati

"""
import numpy as np
class MultipleRidgeRegression:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.coef_ = None
        self.intercept_ = None
        
    def fit(self, Xtrain, Ytrain ):
        Xtrain = np.insert(Xtrain,0,1,axis=1)
        I = np.identity(Xtrain.shape[1])
        result = np.linalg.inv( Xtrain.T @ Xtrain + self.alpha * I) @ Xtrain.T @ Ytrain
        self.coef_ = result[0]
        self.intercept_ = result[1:]
        
    def predict(self , Xtest):
        return (Xtest @ self.coef_) + self.intercept_
    
    def r2_score(self, Ytrue, Ypred):
        ss_total = np.sum((Ytrue - Ytrue.mean()) ** 2)
        ss_res = np.sum((Ytrue - Ypred) ** 2)
        return 1 - (ss_res / ss_total)

