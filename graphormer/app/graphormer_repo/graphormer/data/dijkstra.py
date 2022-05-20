import math
import time
import torch 
import numpy as np

def arg_min(T, S):
    amin = -1
    m = math.inf  # максимальное значение
    for i, t in enumerate(T):
        if t < m and i not in S:
            m = t
            amin = i

    return amin

def prepare_adj(D):
    N = len(D) 
    mask = (torch.ones(N,N, dtype=torch.float) - torch.eye(N,N, dtype=torch.float))*500
    return torch.where(D > 0, D, mask)

def algo_dijk(D):
    D = prepare_adj(D)
    N = len(D)  # число вершин в графе
    T = [math.inf]*N   # последняя строка таблицы
    path = list()
    distance_matrix = np.zeros([N, N])
    path_matrix = np.zeros([N, N])
    for node in range(N):
        v = node     # стартовая вершина (нумерация с нуля)
        S = {v}     # просмотренные вершины
        T = [math.inf]*N   # последняя строка таблицы
        T[v] = 0    # нулевой вес для стартовой вершины
        M = [0]*N   # оптимальные связи между вершинами

        while v != -1:          # цикл, пока не просмотрим все вершины
            for j, dw in enumerate(D[v]):   # перебираем все связанные вершины с вершиной v
                if j not in S:           # если вершина еще не просмотрена
                    w = T[v] + dw
                    if w < T[j]:
                        T[j] = w
                        M[j] = v        # связываем вершину j с вершиной v

            v = arg_min(T, S)            # выбираем следующий узел с наименьшим весом
            if v >= 0:                    # выбрана очередная вершина
                S.add(v)                 # добавляем новую вершину в рассмотрение

    #print(T, M, sep="\n")
        
        distance_matrix[node] = T
        path_matrix[node] = M

        
    return(distance_matrix, path_matrix)

def get_all_edges_dijk(path, i, j):
    # формирование оптимального маршрута:
    start = i
    end = j
    P = [j]
    while end != start:
        end = int(path[i][P[-1]])
        P.append(end)
    return P
