# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import cython
from cython.parallel cimport prange, parallel
cimport numpy
import numpy
import time



def arg_min(T, S):
    cdef int amin = -1
    cdef int m = 510 # максимальное значение
    cdef unsigned int i, j, t
    for i, t in enumerate(T):
        if t < m and i not in S:
            m = t
            amin = i
    return amin


def algo_dijk(D):
    (nrows, ncols) = D.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = D.astype(long, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] distance_matrix = adj_mat_copy
    cdef numpy.ndarray[long, ndim=2, mode='c'] path_matrix = numpy.zeros([n, n], dtype=numpy.int64)
    
    
    cdef unsigned int i, j, k
    cdef int v
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i][j] = 0
            elif distance_matrix[i][j] == 0:
                distance_matrix[i][j] = 510
                
    for node in range(n):
        v = node     # стартовая вершина (нумерация с нуля)
        S = {v}     # просмотренные вершины
        T = [510]*n   # последняя строка таблицы
        T[v] = 0    # нулевой вес для стартовой вершины
        M = [0]*n   # оптимальные связи между вершинами

        while v != -1:          # цикл, пока не просмотрим все вершины
            for j, dw in enumerate(distance_matrix[v]):   # перебираем все связанные вершины с вершиной v
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


def floyd_warshall(adjacency_matrix):
    # print('start floyd_warshall')
    (nrows, ncols) = adjacency_matrix.shape
    assert nrows == ncols
    cdef unsigned int n = nrows

    adj_mat_copy = adjacency_matrix.astype(long, order='C', casting='safe', copy=True)
    assert adj_mat_copy.flags['C_CONTIGUOUS']
    cdef numpy.ndarray[long, ndim=2, mode='c'] M = adj_mat_copy
    cdef numpy.ndarray[long, ndim=2, mode='c'] path = numpy.zeros([n, n], dtype=numpy.int64)

    cdef unsigned int i, j, k
    cdef long M_ij, M_ik, cost_ikkj
    cdef long* M_ptr = &M[0,0]
    cdef long* M_i_ptr
    cdef long* M_k_ptr

    # set unreachable nodes distance to 510
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = 0
            elif M[i][j] == 0:
                M[i][j] = 510

    # floyed algo
    for k in range(n):
        M_k_ptr = M_ptr + n*k
        for i in range(n):
            M_i_ptr = M_ptr + n*i
            M_ik = M_i_ptr[k]
            for j in range(n):
                cost_ikkj = M_ik + M_k_ptr[j]
                M_ij = M_i_ptr[j]
                if M_ij > cost_ikkj:
                    M_i_ptr[j] = cost_ikkj
                    path[i][j] = k

    # set unreachable path to 510
    for i in range(n):
        for j in range(n):
            if M[i][j] >= 510:
                path[i][j] = 510
                M[i][j] = 510
    
    # print('end floyd_warshall')
    return M, path


def get_all_edges(path, i, j):
    cdef unsigned int k = path[i][j]
    if k == 0:
        return []
        # print('end get_all_edges')
    else:
        # print('end get_all_edges')
        return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)
    
# def get_all_edges_dijk(path, i, j):
#     cdef unsigned int k = path[i][j]
#     if k == 0:
#         return []
#         # print('end get_all_edges')
#     else:
#         # print('end get_all_edges')
#         return get_all_edges(path, i, k) + [k] + get_all_edges(path, k, j)

def get_all_edges_dijk(path, i, j):
    # формирование оптимального маршрута:
    cdef unsigned int start = i
    cdef unsigned int end = j
    P = list([j])
    while end != start:
        end = int(path[i][P[-1]])
        P.append(end)
    return P

def gen_edge_input(max_dist, path, edge_feat):
    # print('start gen_edge_input')
    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(long, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']
    
    
    # print('start numpy')
    start_numpy = time.time()
    cdef numpy.ndarray[long, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    end_numpy= time.time()
    # print('end numpy')
    # print('numpy end with time', end_numpy - start_numpy)
    
    cdef unsigned int i, j, k, num_path, cur
    
    # print('start sycle')
    start_sycle = time.time()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = [i] + get_all_edges(path_copy, i, j) + [j]
            # print('type path', type(path))
            # path = [i] + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]
                # edge_fea_all[i, j, k, :] = edge_feat_copy[path[0], path[1], :]
                
    end_sycle = time.time()
    # print('end sycle')
    # print('sycle end with time', end_sycle - start_sycle)
    
    # print('end gen_edge_input')
    return edge_fea_all

def gen_edge_input_dijk(max_dist, path, edge_feat):
    # print('start gen_edge_input')
    (nrows, ncols) = path.shape
    assert nrows == ncols
    cdef unsigned int n = nrows
    cdef unsigned int max_dist_copy = max_dist

    path_copy = path.astype(long, order='C', casting='safe', copy=True)
    edge_feat_copy = edge_feat.astype(long, order='C', casting='safe', copy=True)
    assert path_copy.flags['C_CONTIGUOUS']
    assert edge_feat_copy.flags['C_CONTIGUOUS']
    
    
    # print('start numpy')
    start_numpy = time.time()
    cdef numpy.ndarray[long, ndim=4, mode='c'] edge_fea_all = -1 * numpy.ones([n, n, max_dist_copy, edge_feat.shape[-1]], dtype=numpy.int64)
    end_numpy= time.time()
    # print('end numpy')
    # print('numpy end with time', end_numpy - start_numpy)
    
    cdef unsigned int i, j, k, num_path, cur
    
    # print('start sycle')
    start_sycle = time.time()
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if path_copy[i][j] == 510:
                continue
            path = get_all_edges_dijk(path_copy, i, j)
            # print('type path', type(path))
            # path = [i] + [j]
            num_path = len(path) - 1
            for k in range(num_path):
                edge_fea_all[i, j, k, :] = edge_feat_copy[path[k], path[k+1], :]
                # edge_fea_all[i, j, k, :] = edge_feat_copy[path[0], path[1], :]
                
    end_sycle = time.time()
    # print('end sycle')
    # print('sycle end with time', end_sycle - start_sycle)
    
    # print('end gen_edge_input')
    return edge_fea_all