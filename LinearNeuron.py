import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class LinearNeuron:
    def __init__(self, in_features, λ):
        self.in_features = in_features
        self.λ = 0.1
        self.W = np.zeros((self.in_features + 1, 1))

    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    def predict(self, x):
        xlike = np.ones_like(x)
        x = np.stack((x, xlike), axis=-1).squeeze()
        z = self.W.T @ x
        return z

    def fit(self, X, Y, lr=0.1):
        n = X.shape[0]  # sample counts
        d = X.shape[1]  # dim of features
        if d != self.in_features:
            raise ValueError('X.shape[0] must be self.in_features (%d)' % (self.in_features))

        Y = Y.reshape((1, -1))
        if Y.shape[1] != n:
            raise ValueError('Y.shape[1] must be sample counts (%d)' % (n))

        # initialize params
        self.W = np.zeros((self.in_features + 1, 1))

        loss0 = np.Inf
        epsilon = 1e-6
        iter = 0
        while (True):
            # predict
            rho = self.predict(X)  # 1 x n

            # loss
            loss = (self.W.T.dot(X.T).dot(X).dot(self.W) - 2 * self.W.T.dot(X.T).dot(Y) + self.λ * self.W.T.dot(
                self.W) + Y.T.dot(Y)) / X.shape[0]

            print('iter = %03d, loss = %.6f' % (iter, loss))
            iter = iter + 1

            if np.abs(loss - loss0) < epsilon:
                break

            loss0 = loss

            # error:
            # e = rho - Y  # 1 x n

            # dw,db:
            dw = 2 * (X.T.dot(X).dot(self.W) - X.T.dot(Y) - self.λ * self.W) / X.shape[0]

            self.W = self.W - lr * dw

    def evaluate(self, x, y):
        rho = self.predict(x)
        yhat = np.where(rho >= 0.5, 1, 0)
        err = (yhat != y).astype('float').mean()
        return err


def splitData(x, y, train_size=0.7):
    shuffle_indices = np.random.permutation(x.shape[1])
    x = x[:, shuffle_indices]
    y = y[shuffle_indices]

    split_index = int(x.shape[1] * 0.7)

    x_train = x[:, :split_index]
    y_train = y[:split_index]

    x_test = x[:, split_index:]
    y_test = y[split_index:]

    return x_train, y_train, x_test, y_test


def test1():
    cell = LinearNeuron(in_features=5)
    print('W = ', cell.W)
    print('b = ', cell.b)

    x = np.random.randn(5, 10)
    a = cell.predict(x)
    print(a)


def test2():
    # 生成一组2维随机样本
    n = 200
    x1 = np.random.normal([[1], [1]], 0.8, (2, n))
    x0 = np.random.normal([[-1], [-1]], 0.8, (2, n))
    x = np.hstack((x0, x1))
    y = np.hstack((np.zeros((n,)), np.ones((n,))))
    print(y.shape)

    x_train, y_train, x_test, y_test = splitData(x, y)

    print(y_train.shape)
    print(x_test.shape)

    plt.plot(x_train[0, y_train == 0], x_train[1, y_train == 0], 'b.', alpha=0.8)
    plt.plot(x_train[0, y_train == 1], x_train[1, y_train == 1], 'r.', alpha=0.8)
    plt.axis('equal')
    plt.grid()
    plt.show()

    lr = LinearNeuron(in_features=2)
    lr.fit(x_train, y_train, lr=1)

    # 计算训练误差:
    err = lr.evaluate(x_train, y_train)
    print('training error = %.2f%%' % (err * 100))

    # 计算测试误差
    err = lr.evaluate(x_test, y_test)
    print('test error = %.2f%%' % (err * 100))

    # 绘制分界面
    plt.plot(x_train[0, y_train == 0], x_train[1, y_train == 0], 'b.', alpha=0.8)
    plt.plot(x_train[0, y_train == 1], x_train[1, y_train == 1], 'r.', alpha=0.8)

    xs = np.array([-3, 3])
    ys = -(lr.W[0, 0] * xs + lr.b) / lr.W[1, 0]
    plt.plot(xs, ys, 'k-')
    plt.axis('equal')
    plt.grid()
    plt.show()


def testMNIST():
    df = pd.read_csv('data/MNIST/mnist_test.csv')
    data = np.vstack((df.columns.to_numpy(), df.to_numpy())).astype(np.float32)
    y, x = data[:, 0], data[:, 1:]
    zero_ones = y < 2
    x = x[zero_ones, :].T
    y = y[zero_ones]

    x = 2 * x / 255 - 1

    x_train, y_train, x_test, y_test = splitData(x, y)

    lr = LinearNeuron(in_features=x_train.shape[0])
    lr.fit(x_train, y_train, lr=1)

    # 计算训练误差:
    err = lr.evaluate(x_train, y_train)
    print('training error = %.2f%%' % (err * 100))

    # 计算测试误差
    err = lr.evaluate(x_test, y_test)
    print('test error = %.2f%%' % (err * 100))

    # 绘制权值
    w = lr.W.reshape((28, 28))
    w = (w - w.min()) / (w.max() - w.min())
    plt.imshow(w, cmap='gray')
    plt.show()


if __name__ == '__main__':
    test2()
    # testMNIST()
