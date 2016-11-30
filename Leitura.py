import csv
import numpy as np
from datetime import datetime as dt, timedelta


# arquivo de dados
def read_data(nome_arq):
    # 2 tipos de edicao: Tamanho (em dias) e janela (semanal (5 dias) ou mensal (+-20 dias))
    janela = "diario"
    tamanho = 90
    n_stocks = 5
    # Le csv e joga numa lista de tuplas dados
    with open(nome_arq, 'rb') as f:
        reader = csv.reader(f)
        dados = map(tuple, reader)

    # delimitar tamanho
    data_inicio = dt.strptime("01/01/16", "%m/%d/%y") # isso eh fixo
    delta = timedelta(days=tamanho+1) # delta manipulado pelo tamanho
    end_date = data_inicio + delta  # data final de acordo com o tamanho
    end_date = end_date.strftime('%d/%m/%y')
    # gambiarra com as datas para contar os dias uteis e talz
    datas = []
    for linha in dados[1:]:
        datas.append(linha[0])
    # econtra a data especificada
    for index, item in enumerate(datas):
        if item == end_date:
            dados = dados[1:index+1]
            datas = datas[:index]
            break

    # neste momento o vetor dados esta do tamanho certo.
    # tipo de dados (janela) quando diaria, nao faz nada
    data_igual = []
    dados_mod = []
    # preparando o vetor de dados para contagem
    for index in range(len(dados)-1):
        if dados[index][0] == dados[index+1][0]:
            data_igual.append(dados[index])
        if dados[index][0] != dados[index+1][0]:
            data_igual.append(dados[index])
            dados_mod.append(data_igual[:])
            del data_igual[:]
    dados_fim = []
    cotacoes = []
    i = 0
    subiu = 0
    baixou = 0
    j = 0

    if janela == "semanal":
        while i < len(dados_mod) - 1:
            for index, linha in enumerate(dados_mod):
                if index == 0 or index % 5 != 0:
                    if linha[0][13] == "SUBIU":
                        subiu = subiu + 1
                    else:
                        baixou = baixou + 1
                    for x in range(len(linha)):
                        cotacoes.append(linha[x][11])
                    j = j + 1
                    i = i + 1
                if index != 0 and index%5 == 0:
                    j = 0
                    if subiu > baixou:
                        rotulo_semana = "SUBIU"
                    else:
                        rotulo_semana = "BAIXOU"
                    cotacoes.append(rotulo_semana)
                    dados_fim.append(cotacoes[:])
                    del cotacoes[:]
                    baixou = 0
                    subiu = 0
                    i = i + 1

    if janela == "mensal":
        jan, fev, mar, abr, mai, jun = 0, 0, 0, 0, 0, 0
        for index, linha in enumerate(datas):
            if '/01/' in linha:
                jan = index
            if '/02/' in linha:
                fev = index
            if '/03/' in linha:
                mar = index
            if '/04/' in linha:
                abr = index
            if '/05/' in linha:
                mai = index
            if '/06/' in linha:
                jun = index
        meses = jan, fev, mar, abr, mai, jun
        m = 0
        inicio = 0
        for mes in meses:
            if mes != 0:
                for index, linha in enumerate(dados[inicio:mes+1]):
                    if linha[13] == "SUBIU":
                        subiu = subiu + 1
                    else:
                        baixou = baixou + 1
                    cotacoes.append(linha[11])
                if subiu > baixou:
                    rotulo_semana = "SUBIU"
                else:
                    rotulo_semana = "BAIXOU"
                cotacoes.append(rotulo_semana)
                dados_fim.append(cotacoes[:])
                del cotacoes[:]
                baixou = 0
                subiu = 0
                inicio = mes

    if janela == "diario":
        del data_igual[:]
        for index in range(len(dados) - 1):
            if dados[index][0] == dados[index + 1][0]:
                data_igual.append(dados[index][11])
            if dados[index][0] != dados[index + 1][0]:
                data_igual.append(dados[index][11])
                data_igual.append(dados[index][13])
                dados_fim.append(data_igual[:])
                del data_igual[:]

    del dados_fim[0]
    final_list = []
    for index in range(len(dados_fim)):
        if len(dados_fim[index]) == n_stocks + 1:
            dados_fim[index].insert(-1, 1)
            final_list.append(dados_fim[index])

    return (np.matrix(final_list)[:, :-1], np.matrix(final_list)[:, -1])


# ao final dados fim contem:
# diario: Um vetor de tuplas por dia (1 fechamento, 1 rotulo)
# semanal: Um vetor de vetores de tuplas por semana ((5 dias x Bolsas) fechamentos, 1 rotulo)
# mensal: Um vetor de vetores de tuplas ((+- 21 dias x bolsas) fechamentos, 1 rotulo)
