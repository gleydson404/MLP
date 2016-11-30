# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt

X = np.matrix([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1]])
Y = np.matrix([0, 1, 1, 0])


n_neuro_hidden_layer = 3
n_inputs = 2
n_outs = 2
n_epoc = 10000
alfa = 0.5
epsilon = 0.01

a = np.matrix(np.random.random((n_neuro_hidden_layer, n_inputs + 1)))
b = np.matrix(np.random.random((n_outs, n_neuro_hidden_layer + 1)))

# a = np.matrix([[0.1, -0.1, -0.1], [0.1, 0.1, -0.1], [-0.1, -0.1, 0.1]])
# b = np.matrix([[0.1, 0.0, 0.1, -0.1], [-0.1, 0.1, -0.1, 0.1]])


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def d_sigmoin(x):
    return sigmoid(x) * (1 - sigmoid(x))


def error(y, yd):
    return y - yd


def square_error(err_vec):
    sqr_err = [(i ** 2) * 0.5 for i in err_vec]
    return [np.sum(i) for i in sqr_err]


def mean_square_error(square_error):
    return np.mean(square_error)


def encod(n_instances, n_outs, y):
    enc = np.zeros((n_instances, n_outs))
    for i, item in enumerate(y.tolist()[0]):

        enc[i][item] = 1

    return enc


def feed_forward_old(input, yd, a, b):  # weights should be transpose
    z = []
    y = []
    zin = []
    yin = []
    for index in range(len(a)):
        summary_l1 = input * a[:, index]
        zin.append(summary_l1)
        z.append(sigmoid(summary_l1))
    z.append(1)

    for k in range(len(b.T)):
        summary_l2 = 0
        for i in range(len(b)):
            summary_l2 += z[i] * b.T[k].item(i)
        yin.append(summary_l2)
        y.append(sigmoid(summary_l2))

    return z, zin, yin, y, error(y, yd)


def train_old(x, yd, a, b, epsilon, alfa, n_epoc):
    epoc = 0
    mse = 1
    epoc_list = []
    mse_list = []
    while mse >= epsilon and n_epoc >= epoc:
        epoc_erros = []
        epoc += 1
        for h, item in enumerate(x.tolist()):
            gradient_b = []
            gradient_a = []
            gradient_sqr_err = []
            z, zin, yin, y, erro = feed_forward_old(item, yd[h], a.T, b.T)

            for k in range(n_outs):
                for i in range(n_neuro_hidden_layer + 1):
                    gradient_b.append(erro[k] * d_sigmoin(yin[k]) * z[i])

            for k in range(n_outs):
                for i in range(n_neuro_hidden_layer):
                    gradient_sqr_err.append(erro[k] * d_sigmoin(yin[k]) * b[k].item(i))

            m_gradient_sqr_err = np.reshape(np.matrix(gradient_sqr_err), (n_outs, n_neuro_hidden_layer))
            gradient_sqr_err_vec = np.sum(m_gradient_sqr_err, axis=0)

            for i in range(n_neuro_hidden_layer):
                for j in range(n_inputs + 1):
                    gradient_a.append(gradient_sqr_err_vec.item(i) *
                                      d_sigmoin(zin[i].item(0)) * item[j])

            m_grad_b = np.reshape(np.matrix(gradient_b), (n_outs, n_neuro_hidden_layer + 1))
            m_grad_a = np.reshape(np.matrix(gradient_a), (n_inputs + 1, n_neuro_hidden_layer))

            for i in range(n_outs):
                for j in range(n_neuro_hidden_layer + 1):
                    b[i, j] = b[i, j] - (alfa * m_grad_b[i, j])

            for i in range(n_neuro_hidden_layer):
                for j in range(n_inputs + 1):
                    a[i, j] = a[i, j] - (alfa * m_grad_a[i, j])

            z, zin, yin, y, erro1 = feed_forward_old(item, yd[h], a.T, b.T)
            epoc_erros.append(erro1)

        sqr_err = square_error(epoc_erros)
        mse = mean_square_error(sqr_err)
        print("epoca", epoc)
        print("erro normal ", erro1)
        print("Mse", mse)
        mse_list.append(mse)
        epoc_list.append(epoc)

    plt.xlabel("Epoca")
    plt.ylabel("Erro Quadratico")
    plt.plot(epoc_list, mse_list)
    plt.show()
    return a, b


def feed_forward(input, yd, a, b):  # weights should be transpose
    z = []
    y = []
    zin = []
    yin = []
    for index in range(n_neuro_hidden_layer):
        summary_l1 = input * a[:, index]
        zin.append(summary_l1)
        z.append(sigmoid(summary_l1))
    z.append(1)

    for k in range(len(b.T)):
        summary_l2 = 0
        for i in range(len(z)):
            summary_l2 += z[i] * b.T[k].item(i)
        yin.append(summary_l2)
        y.append(sigmoid(summary_l2))

    return z, zin, yin, y, error(y, yd)


def train(x, yd, a, b, epsilon, alfa, n_epoc):
    epoc = 0
    mse = 1
    epoc_list = []
    mse_list = []
    while mse >= epsilon and n_epoc >= epoc:
        epoc_erros = []
        epoc += 1
        for h, item in enumerate(x.tolist()):
            gradient_b = []
            gradient_a = []
            gradient_sqr_err = []
            z, zin, yin, y, erro = feed_forward(item, yd[h], a.T, b.T)

            for k in range(n_outs):
                for i in range(n_neuro_hidden_layer + 1):
                    gradient_b.append(erro[k] * d_sigmoin(yin[k]) * z[i])

            for k in range(n_outs):
                for i in range(n_neuro_hidden_layer):
                    gradient_sqr_err.append(erro[k] * d_sigmoin(yin[k]) * b[k].item(i))

            m_gradient_sqr_err = np.reshape(np.matrix(gradient_sqr_err), (n_outs, n_neuro_hidden_layer))
            gradient_sqr_err_vec = np.sum(m_gradient_sqr_err, axis=0)

            for i in range(n_neuro_hidden_layer):
                for j in range(n_inputs + 1):
                    gradient_a.append(gradient_sqr_err_vec.item(i) *
                                      d_sigmoin(zin[i].item(0)) * item[j])

            m_grad_b = np.reshape(np.matrix(gradient_b), (n_outs, n_neuro_hidden_layer + 1))
            m_grad_a = np.reshape(np.matrix(gradient_a), (n_neuro_hidden_layer, n_inputs + 1))

            for i in range(n_outs):
                for j in range(n_neuro_hidden_layer + 1):
                    b[i, j] = b[i, j] - (alfa * m_grad_b[i, j])

            for i in range(n_neuro_hidden_layer):
                for j in range(n_inputs + 1):
                    a[i, j] = a[i, j] - (alfa * m_grad_a[i, j])

            z, zin, yin, y, erro1 = feed_forward(item, yd[h], a.T, b.T)

            # print("erro normal ", erro1)
            epoc_erros.append(erro1)
        sqr_err = square_error(epoc_erros)
        mse = mean_square_error(sqr_err)
        print("epoca", epoc)
        print("Mse", mse)
        mse_list.append(mse)
        epoc_list.append(epoc)

    plt.xlabel("Epoca")
    plt.ylabel("Erro Quadratico")
    plt.plot(epoc_list, mse_list)
    plt.show()
    return a, b



Yd = encod(len(X), n_outs, Y)
# print(feed_forward(X[0], Yd[0], a.T, b.T))
train(X, Yd, a, b, epsilon, alfa, n_epoc)
