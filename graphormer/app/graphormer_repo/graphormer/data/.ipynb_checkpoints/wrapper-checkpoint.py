# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
import numpy as np
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4mv2_pyg import PygPCQM4Mv2Dataset
from functools import lru_cache
import pyximport
import torch.distributed as dist
import time
pyximport.install(setup_args={"include_dirs": np.get_include()})
from . import algos
from . import dijkstra
from . import bfs

@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


@torch.jit.script
def convert_to_single_emb(x, offset: int = 512):
    feature_num = x.size(1) if len(x.size()) > 1 else 1
    feature_offset = 1 + torch.arange(0, feature_num * offset, offset, dtype=torch.long)
    x = x + feature_offset
    return x


def preprocess_item(item, short_path_mode = 'dijkstra'):
    
    start = time.time()
    edge_attr, edge_index, x = item.edge_attr, item.edge_index, item.x
    N = x.size(0)
    x = convert_to_single_emb(x)

    # node adj matrix [N, N] bool
    adj = torch.zeros([N, N], dtype=torch.bool)
    adj[edge_index[0, :], edge_index[1, :]] = True

    # edge feature here
    if len(edge_attr.size()) == 1:
        edge_attr = edge_attr[:, None]
    attn_edge_type = torch.zeros([N, N, edge_attr.size(-1)], dtype=torch.long)
    attn_edge_type[edge_index[0, :], edge_index[1, :]] = (
        convert_to_single_emb(edge_attr) + 1
    )
    
    start_floyd = time.time()
    if short_path_mode == 'dijkstra':
        # print('start find path floyd')
        start_algo= time.time()
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        end_algo = time.time()
        # print('floyd end with time', end_algo - start_algo)
        
        max_dist = np.amax(shortest_path_result)
        
        # print('start gen_edge_input floyd')
        start_gen_edge_input= time.time()
        edge_input = algos.gen_edge_input(max_dist, path, attn_edge_type.numpy())
        end_gen_edge_input = time.time()
        # print('gen_edge_input floyd end with time', end_gen_edge_input - start_gen_edge_input)
        dijkstra
    elif short_path_mode == 'floyd':
        # print('start find path dijkstra')
        start_algo= time.time()
        shortest_path_result, path = algos.algo_dijk(adj.numpy())
        end_algo = time.time()
        # print('dijkstra end with time', end_algo - start_algo)
        max_dist = np.amax(shortest_path_result)
        
        # print('start gen_edge_input dijkstra')
        start_gen_edge_input= time.time()
        edge_input = algos.gen_edge_input_dijk(max_dist, path, attn_edge_type.numpy())
        end_gen_edge_input = time.time()
        # print('gen_edge_input dijkstra end with time', end_gen_edge_input - start_gen_edge_input)
    elif short_path_mode == 'bfs':
        # print('start find path bfs')
        start_algo= time.time()
        shortest_path_result, path = algos.floyd_warshall(adj.numpy())
        end_algo = time.time()
        # print('bfs end with time', end_algo - start_algo)
        
        max_dist = np.amax(shortest_path_result)
        
        # print('start gen_edge_input bfs')
        start_gen_edge_input= time.time()
        edge_input = algos.gen_edge_input_dijk(max_dist, path, attn_edge_type.numpy())
        end_gen_edge_input = time.time()
        # print('gen_edge_input bfs end with time', end_gen_edge_input - start_gen_edge_input)
    spatial_pos = torch.from_numpy((shortest_path_result)).long()
    attn_bias = torch.zeros([N + 1, N + 1], dtype=torch.float)  # with graph token

    # combine
    item.x = x
    item.attn_bias = attn_bias
    item.attn_edge_type = attn_edge_type
    item.spatial_pos = spatial_pos
    item.in_degree = adj.long().sum(dim=1).view(-1)
    item.out_degree = item.in_degree  # for undirected graph
    item.edge_input = torch.from_numpy(edge_input).long()
    
    end = time.time()
    return item


class MyPygPCQM4MDataset(PygPCQM4Mv2Dataset):
    def download(self):
        super(MyPygPCQM4MDataset, self).download()

    def process(self):
        super(MyPygPCQM4MDataset, self).process()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        return preprocess_item(item)


class MyPygGraphPropPredDataset(PygGraphPropPredDataset):
    def download(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).download()
        dist.barrier()

    def process(self):
        if dist.get_rank() == 0:
            super(MyPygGraphPropPredDataset, self).process()
        dist.barrier()

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        item = self.get(self.indices()[idx])
        item.idx = idx
        item.y = item.y.reshape(-1)
        # print(item)
        return preprocess_item(item)
