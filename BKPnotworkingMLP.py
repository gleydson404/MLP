# -*- coding: utf-8 -*-
import numpy as np
import math
X = np.matrix([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1]])
Y = np.matrix([0, 1, 1, 0])

n_neuro_hidden_layer = 3
n_inputs = 2
n_outs = 2
n_epoc = 1000
alfa = 0.001
epsilon = 0.001

a = np.random.random_sample((n_neuro_hidden_layer, n_inputs + 1))
b = np.random.random_sample((n_outs, n_neuro_hidden_layer + 1))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def deriv_sigmoid(x):
    vectorize_sigmoid = np.vectorize(sigmoid)
    ones = np.ones(x.shape)
    return np.multiply(vectorize_sigmoid(x), (ones - vectorize_sigmoid(x)))


def feed(X, a, b):
    # vectozize aplica a função para cada elemento da matriz
    vectozize_sigmoid = np.vectorize(sigmoid)
    zin = X * a.T
    z = vectozize_sigmoid(zin)
    bias_z = np.ones((X.shape[0], 1), dtype=z.dtype)
    z = np.hstack((z, bias_z))
    yin = z * b.T
    y = vectozize_sigmoid(yin)
    return y, yin, z, zin


def encod(n_instances, n_outs):
    enc = np.zeros((n_instances, n_outs))
    for i, item in enumerate(Y.tolist()[0]):

        enc[i][item] = 1

    return enc


def calc_local_error(out, desired_out):
    return out - desired_out


def calc_global_error(out, desired_out):
    return np.sum(out - desired_out, axis=0)


def calc_mean_square_error(out, desired_out):
    error = out - desired_out
    vectorize_square = np.vectorize(math.pow)
    error = vectorize_square(error, 2)
    error = error.sum(axis=0)
    return error / (2 * (X.shape[0]))


def norm_error(error):
    sum_error = 0
    for i in range(len(error)):
        sum_error += math.pow(error.item(i), 2)
    return sum_error


def train(X, Y, a, b, epsilon, alfa, n_epoc):
    out, yin, z, zin = feed(X, a, b)
    epoc = 0
    coded_desired_out = encod(X.shape[0], n_outs)
    erro_local = calc_local_error(out, coded_desired_out)
    error_globl = calc_global_error(out, coded_desired_out)
    mse = norm_error(calc_mean_square_error(out, coded_desired_out))
    while mse >= epsilon and n_epoc >= epoc:
        epoc += 1
        import ipdb; ipdb.set_trace()
        deriv_bki = (np.multiply(erro_local, deriv_sigmoid(yin)).T * z)
        deriv_aij = np.multiply((np.multiply(error_globl, deriv_sigmoid(yin)) * b),
                                deriv_sigmoid(zin).T * X)
        b = (b - alfa * deriv_bki)
        a = (a - alfa * deriv_aij)
        out, yin, z, zin = feed(X, a, b)
        b = erro * deriv_sigmoid(out) * out
        a = b * deriv_sigmoid()
        erro = calc_mean_square_error(out, coded_desired_out)
        mse = norm_error(erro)
        print("mse", mse)

print(feed(X[0], a, b))
