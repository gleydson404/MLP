# -*- coding: utf-8 -*-
import numpy as np
import math
import matplotlib.pyplot as plt
from collections import Counter
from Leitura import read_data

# X = np.matrix([[1, 1, 1], [1, 0, 1], [0, 1, 1], [0, 0, 1]])
# Y = np.matrix([0, 1, 1, 0])
X, Y = read_data("CEN1_DATASET1.csv")

X = X.astype(np.float)
history_outs = [[0, 0, 0]]

n_neuro_hidden_layer = 10
n_inputs = 5
n_outs = 3
n_epoc = 10000
alfa = 0.5
epsilon = 0.01

# a = np.matrix(0.1 * np.random.random((n_neuro_hidden_layer, n_inputs + 1)) - 0.1)
# b = np.matrix(0.1 * np.random.random((n_outs, n_neuro_hidden_layer + 1)) - 0.1)


a = np.matrix(np.random.random((n_neuro_hidden_layer, n_inputs + 1 + 3)))
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


def encod(y):
    y_encod = []
    for item in y:
        if item == "BAIXOU":
            y_encod.append([1, 0, 0])
        if item == "ESTAVEL":
            y_encod.append([0, 1, 0])
        if item == "SUBIU":
            y_encod.append([0, 0, 1])

    return np.array(y_encod)


def clean_str(string):
    return string.replace("[", "")\
                 .replace("]", "")\
                 .replace(" ", "")\
                 .replace(".", "")


def decod(y):
    if y == '100':
        return "BAIXOU"
    if y == '010':
        return "ESTAVEL"
    if y == '001]':
        return "SUBIU"


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

    history_outs.append(y)

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

            item.extend(history_outs[-1])
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
                for j in range(n_inputs + 1 + 3):
                    gradient_a.append(gradient_sqr_err_vec.item(i) *
                                      d_sigmoin(zin[i].item(0)) * item[j])

            m_grad_b = np.reshape(np.matrix(gradient_b), (n_outs, n_neuro_hidden_layer + 1))
            m_grad_a = np.reshape(np.matrix(gradient_a), (n_neuro_hidden_layer, n_inputs + 1 + 3))

            for i in range(n_outs):
                for j in range(n_neuro_hidden_layer + 1):
                    b[i, j] = b[i, j] - (alfa * m_grad_b[i, j])

            for i in range(n_neuro_hidden_layer):
                for j in range(n_inputs + 1 + 3):
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


def predict(a, b, input):
    z = []
    y = []
    zin = []
    yin = []

    input.extend(history_outs[-1])

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
    history_outs.append(y)
    return np.round(y)


def slice_data(x, y, qtd_percent):
    x_train = []
    y_train = []
    y_test = []
    x_test = []

    for index in range(int(len(x) * qtd_percent)):
        x_train.append(x[index].tolist()[0])
        y_train.append(y[index].tolist()[0])

    for index in range(int(len(x) * (1 - qtd_percent))):
        x_test.append(x[index].tolist()[0])
        y_test.append(y[index].tolist()[0])

    return x_train, y_train, x_test, y_test


X_train, Y_train, X_test, Y_test = slice_data(X, Y, 0.7)

X_train = np.matrix(X_train)
Y_train = np.matrix(Y_train)
X_test = np.matrix(X_test)
Y_test = np.matrix(Y_test)


Yd = encod(Y_train)
# Treina a rede aqui
# A, B = train(X_train, Yd, a, b, epsilon, alfa, n_epoc)
# np.savetxt("weightsA0.5_recorrent", A, delimiter=',')
# np.savetxt("weightsB0.5_recorrent", B, delimiter=',')

# carrega os pesos dos arquivos lembra de mudar o nome pra o arquivo que você salvou
# pra só rodar, descomentar da proxima linha ate o final
A = np.matrix(np.loadtxt("weightsA0.5_recorrent", delimiter=','))
B = np.matrix(np.loadtxt("weightsB0.5_recorrent", delimiter=','))

# print(predict(A.T, B.T, X_test[0]))
# print(Y_test[0])

# Executa  a rede com os cenarios de testes
results_list = []
alta = []
media = []
baixa = []
for i, item in enumerate(X_test):
    out = predict(A.T, B.T, item.tolist()[0])
    out = decod(clean_str(str(out)))
    if out == Y_test[i].tolist()[0][0]:
        results_list.append("acertou")
        if out == "BAIXOU":
            baixa.append(1)
        elif out == "ESTAVEL":
            media.append(1)
        elif out == "SUBIU":
            alta.append(1)
    else:
        results_list.append("errou")

print("quantidade de testes", len(results_list))
counter = Counter(results_list)
print(counter)
print("alta", np.sum(alta))
print("media", np.sum(media))
print("baixa", np.sum(baixa))
