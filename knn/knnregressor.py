"""
K Nearest Neighbors Regressor implemented from scratch using Python.
Author: Pruthil Prajapati

"""
import numpy as np
import statistics as st

class KnnRegressor:
    def __init__(self, k=5):
        self.n_neighbors = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted yet. Please call 'fit' with training data before predicting.")
        y_pred = []
        for x in X_test:

            distances = np.linalg.norm(self.X_train - x, axis=1)
            dist_with_idx = list(zip(range(len(distances)), distances))
            sorted_index = sorted(dist_with_idx, key=lambda item: item[1])[:self.n_neighbors]
            n_indices = [item[0] for item in sorted_index]
            y_values = self.y_train[n_indices]
            predicted_value = np.mean(y_values)
            y_pred.append(predicted_value)

        return np.array(y_pred)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        ssr = np.sum((y_test - y_pred)**2)
        sst = np.sum((y_test - np.mean(y_test))**2)
        r2 = 1 - (ssr / sst)
        return r2
    
# knnreg = KnnRegressor(5)
# knnreg.fit(xtrain,ytrain)
# y_pred = knnreg.predict(xtest)
# knnreg.score(xtest,ytest)