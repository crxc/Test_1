# This is a sample Python script.
import random

import matplotlib.pyplot as plt
# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from sklearn.model_selection import KFold, train_test_split
# import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib as mlt
from LinearNeuron import *

num = 100
dim = 1
split_num = 5
# 正则化项
λ = 0.1


def getmin_index(list_error):
    global min_index
    min = np.Infinity
    for index, error in enumerate(list_error):
        if error < min:
            min = error
            min_index = index
    return min_index
    pass


def test1():
    data = Data()
    data.create_data()
    splits = data.divide_data(split_num)
    list_neuron = []
    list_w = []
    list_train_error = []
    list_test_error = []
    for train, test in splits.split(data.train_set_x):
        neuron = LinearNeuron(dim, λ)
        list_neuron.append(neuron)
        neuron.fit(data.train_set_x[train], data.train_set_y[train])
        error = neuron.evaluate(data.train_set_x[test], data.train_set_y[test])
        list_train_error.append(error)
        list_w.append(neuron.W)
    for neuron in list_neuron:
        list_test_error.append(neuron.evaluate(data.test_set_x, data.test_set_y))
    index = getmin_index(list_test_error)
    model = Model()
    model.lsm(data.x, data.y)
    point1 = [0, 100]
    point2 = [model.w.T.dot(np.array([0, 1]).reshape(2, 1))[0][0],
              model.w.T.dot(np.array([100, 1]).reshape(2, 1))[0][0]]
    fig, ax = plt.subplots()
    ax.plot(point1, point2, color="blue")

    # model.GD(data.x, data.y)
    # print(data.x.shape)
    ax.scatter(data.x, data.y)
    point1 = [0, 100]
    point2 = [list_w[index].T.dot(np.array([0, 1]).reshape(2, 1))[0][0],
              list_w[index].T.dot(np.array([100, 1]).reshape(2, 1))[0][0]]
    ax.plot(point1, point2, color="red")
    plt.show()


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    data = Data()
    data.create_data()
    splits = data.divide_data(split_num)
    model = Model()
    list_w = []
    list_train_error = []
    for train, test in splits.split(data.train_set_x):
        model.GD(data.train_set_x[train], data.train_set_y[train])
        error = model.get_loss(data.train_set_x[test], data.train_set_y[test], model.w)
        list_train_error.append(error)
        list_w.append(model.w)
    print(list_w)
    print(list_train_error)
    list_error_test = []
    for w in list_w:
        list_error_test.append(model.get_loss(data.test_set_x, data.test_set_y, w))
    index = getmin_index(list_error_test)
    print(list_train_error[index])
    print(list_error_test[index])
    print(list_error_test)

    model.lsm(data.x, data.y)
    point1 = [0, 100]
    point2 = [model.w.T.dot(np.array([0, 1]).reshape(2, 1))[0][0],
              model.w.T.dot(np.array([100, 1]).reshape(2, 1))[0][0]]
    fig, ax = plt.subplots()
    ax.plot(point1, point2, color="blue")

    # model.GD(data.x, data.y)
    # print(data.x.shape)
    ax.scatter(data.x, data.y)
    point1 = [0, 100]
    point2 = [list_w[index].T.dot(np.array([0, 1]).reshape(2, 1))[0][0],
              list_w[index].T.dot(np.array([100, 1]).reshape(2, 1))[0][0]]
    ax.plot(point1, point2, color="red")
    plt.show()


def question2():
    data = Data()
    data.create_data2()
    splits = data.divide_data(split_num)
    model = Model()


class Data:
    def create_data(self):
        self.x = np.random.uniform(10, 100, (num, dim))
        self.y = 2 * self.x + 1 + np.random.normal(0, 10, (num, dim))

    def divide_data(self, s):
        self.train_set_x, self.test_set_x, self.train_set_y, self.test_set_y = train_test_split(self.x, self.y,
                                                                                                test_size=0.2)
        splits = KFold(n_splits=s, shuffle=True)
        return splits

    def create_data2(self):
        self.x = np.random.uniform(10, 100, (num, dim))
        self.y = self.x ** 3 + 2 * self.x ** 2 + self.x - 1 + np.random.normal(0, 20, (num, dim))
        pass


class Model:
    # y=w*x+b
    α = 0.0002  # 学习速率
    ε = 0.0001
    N_SPLITS = 50

    def lsm(self, X, Y):
        xlike = np.ones_like(X)
        X = np.stack((X, xlike), axis=-1).squeeze()
        self.w = np.linalg.inv(X.T.dot(X) - λ * np.identity(dim + 1)).dot(X.T).dot(Y)
        print(X.shape)
        print(self.w)

    def distance(self, w1, w2):
        return (w2 - w1).T.dot(w2 - w1)

    def GD(self, X, Y):
        global x, y
        xlike = np.ones_like(X)
        X = np.stack((X, xlike), axis=-1).squeeze()
        self.w = np.array([0.0, 0.0]).reshape((2, 1))
        k = 0
        while (True):
            w_old = self.w.copy()
            splits = KFold(n_splits=self.N_SPLITS, shuffle=True)
            for train_index, test_index in splits.split(X):
                x, y = X[test_index], Y[test_index]
                break
            self.w -= (self.α * 2 * (x.T.dot(x).dot(self.w) - x.T.dot(y) - λ * self.w)) / (num / self.N_SPLITS)
            k += 1
            distance = self.distance(w_old, self.w)
            if distance < self.ε:
                print("距离" + str(distance))
                break
        print("循环了" + str(k) + "次")
        print(self.w)

    def get_loss(self, X, Y, W):
        xlike = np.ones_like(X)
        X = np.stack((X, xlike), axis=-1).squeeze()
        # t_ = (W.T.dot(X.T) - Y.T)
        # return (t_.dot(t_.T) + λ * W.T.dot(W)) / X.shape[0]
        return (W.T.dot(X.T).dot(X).dot(W) - 2 * W.T.dot(X.T).dot(Y) + λ * W.T.dot(W) + Y.T.dot(Y)) / X.shape[0]


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    np.seterr(invalid='ignore')
    # print_hi('PyCharm')
    test1()