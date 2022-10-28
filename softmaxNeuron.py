import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SoftMaxNeuron:
    def __init__(self, in_features):
        self.in_features = in_features
        self.W = np.zeros((self.in_features, 1))
        self.b = 0

    def softmax(z):
        shiftz = z - np.max(z)
        exps = np.exp(shiftz)
        return exps / np.sum(exps)

    def softmax_layer(x, W):
        logits = W.dot(x)
        return SoftMaxNeuron.softmax(logits)

    def predict(self, x):
        z = self.W.T @ x + self.b
        a = SoftMaxNeuron.softmax(z)
        return a

    def fit(self, X, Y, lr=0.1):
        n = X.shape[0]  # sample counts
        d = X.shape[1]  # dim of features
        if d != self.in_features:
            raise ValueError('X.shape[0] must be self.in_features (%d)' % (self.in_features))

        Y = Y.reshape((1, -1))
        if Y.shape[1] != n:
            raise ValueError('Y.shape[1] must be sample counts (%d)' % (n))

        # initialize params
        self.W = np.zeros((self.in_features, 1))
        self.b = 0

        loss0 = np.Inf
        epsilon = 1e-6
        iter = 0
        while (True):
            # predict
            rho = self.predict(X)  # 1 x n

            # loss
            loss = -(np.log(rho[Y == 1]).sum() + np.log(1 - rho[Y == 0]).sum()) / n

            print('iter = %03d, loss = %.6f' % (iter, loss))
            iter = iter + 1

            if np.abs(loss - loss0) < epsilon:
                break

            loss0 = loss

            # error:
            e = rho - Y  # 1 x n

            # dw,db:
            dw = X @ e.T / n
            db = e.mean()

            self.W = self.W - lr * dw
            self.b = self.b - lr * db

    def evaluate(self, x, y):
        rho = self.predict(x)
        yhat = np.where(rho >= 0.5, 1, 0)
        err = (yhat != y).astype('float').mean()
        return err
