import numpy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class SoftMaxNeuron:
    def __init__(self, in_features, K):
        self.in_features = in_features
        self.K = K
        self.W = np.zeros((K, in_features + 1))

    def softmax(z):
        shiftz = z - np.max(z,axis=0)
        exps = np.exp(shiftz)
        return exps / np.sum(exps, axis=0)

    def softmax_layer(x, W):
        logits = W.dot(x)
        return SoftMaxNeuron.softmax(logits)

    def predict(self, x):
        xlike = np.ones(shape=(x.shape[0], 1))
        x = np.c_[x, xlike]
        z = self.W @ x.T
        a = SoftMaxNeuron.softmax(z)  # a为概率矩阵
        return a

    def fit(self, X, Y, lr=10):
        n = X.shape[0]  # sample counts
        d = X.shape[1]  # dim of features
        k = self.K
        if d != self.in_features:
            raise ValueError('X.shape[0] must be self.in_features (%d)' % (self.in_features))

        Y = Y.reshape((1, -1))
        if Y.shape[1] != n:
            raise ValueError('Y.shape[1] must be sample counts (%d)' % (n))

        # initialize params
        self.W = np.zeros((self.K, self.in_features + 1))

        loss0 = np.Inf
        epsilon = 1e-3
        iter = 0
        while (True):
            # predict
            rho = self.predict(X)  # 1 x n

            # loss
            loss = 0
            for i in range(n):
                y = np.zeros(shape=(k,))
                y[int(Y[0, i])] = 1
                p = rho[:, i]
                loss += y.T @ numpy.log(p+0.0000000000001)
                pass
            loss = -loss / n
            if loss > loss0:
                lr /= 1.1
            print('iter = %03d, loss = %.6f' % (iter, loss))
            iter = iter + 1

            if np.abs(loss - loss0) < epsilon:
                break

            loss0 = loss
            y = Y.squeeze().astype(int)
            Y_ = np.eye(10)[y.T].T
            # error:
            e = rho - Y  # 1 x n

            xlike = np.ones(shape=(X.shape[0], 1))
            x = np.c_[X, xlike]
            # dw,db:
            dw = (rho - Y_) @ x / n

            self.W = self.W - lr * dw

    def evaluate(self, x, y):
        rho = self.predict(x)
        yhat = np.argmax(rho, axis=0)
        err = (yhat == y).astype('float').mean()
        return err
