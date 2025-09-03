"""
K Nearest Neighbors implemented from scratch using Python.
Author: Pruthil Prajapati

"""
import numpy as np
import statistics as st

class Knn:

    def __init__(self, k=5):
        self.n_neighbors = k
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        if self.X_train is None or self.y_train is None:
            raise ValueError("Model has not been fitted. Please call 'fit' with training data before predicting.")
        y_pred = []
        idx = np.arange(0, self.X_train.shape[0])
        for x in X_test:
            distances = np.linalg.norm(self.X_train - x, axis=1)
            dist_with_indices = sorted(list(zip(idx, distances)), key=lambda x: x[1])[:self.n_neighbors]
            indices = [i[0] for i in dist_with_indices]
            values_Y = self.y_train[indices]
            most_common = st.mode(values_Y)
            y_pred.append(most_common)
        return np.array(y_pred)

    def score(self, X_test, y_test):
        y_pred = self.predict(X_test)
        accuracy = np.mean(y_pred == y_test)
        return accuracy

# knn = Knn(5)
# knn.fit(xtrain,ytrain)
# y_pred = knn.predict(xtest)
# knn.score(xtest,ytest)