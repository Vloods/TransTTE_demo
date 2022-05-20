# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch


def pad_1d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen], dtype=x.dtype)
        new_x[:xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_2d_unsqueeze(x, padlen):
    x = x + 1  # pad id = 0
    xlen, xdim = x.size()
    if xlen < padlen:
        new_x = x.new_zeros([padlen, xdim], dtype=x.dtype)
        new_x[:xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_attn_bias_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype).fill_(float("-inf"))
        new_x[:xlen, :xlen] = x
        new_x[xlen:, :xlen] = 0
        x = new_x
    return x.unsqueeze(0)


def pad_edge_type_unsqueeze(x, padlen):
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen, x.size(-1)], dtype=x.dtype)
        new_x[:xlen, :xlen, :] = x
        x = new_x
    return x.unsqueeze(0)


def pad_spatial_pos_unsqueeze(x, padlen):
    x = x + 1
    xlen = x.size(0)
    if xlen < padlen:
        new_x = x.new_zeros([padlen, padlen], dtype=x.dtype)
        new_x[:xlen, :xlen] = x
        x = new_x
    return x.unsqueeze(0)


def pad_3d_unsqueeze(x, padlen1, padlen2, padlen3):
    x = x + 1
    xlen1, xlen2, xlen3, xlen4 = x.size()
    if xlen1 < padlen1 or xlen2 < padlen2 or xlen3 < padlen3:
        new_x = x.new_zeros([padlen1, padlen2, padlen3, xlen4], dtype=x.dtype)
        new_x[:xlen1, :xlen2, :xlen3, :] = x
        x = new_x
    return x.unsqueeze(0)


# def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
#     items = [item for item in items if item is not None and item.x.size(0) <= max_node]
#     items = [
#         (
#             item.idx,
#             item.attn_bias,
#             item.attn_edge_type,
#             item.spatial_pos,
#             item.in_degree,
#             item.out_degree,
#             item.x,
#             item.edge_input[:, :, :multi_hop_max_dist, :],
#             item.y,
#         )
#         for item in items
#     ]
#     (
#         idxs,
#         attn_biases,
#         attn_edge_types,
#         spatial_poses,
#         in_degrees,
#         out_degrees,
#         xs,
#         edge_inputs,
#         ys,
#     ) = zip(*items)

#     for idx, _ in enumerate(attn_biases):
#         attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float("-inf")
#     max_node_num = max(i.size(0) for i in xs)
#     max_dist = max(i.size(-2) for i in edge_inputs)
#     y = torch.cat(ys)
#     x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
#     edge_input = torch.cat(
#         [pad_3d_unsqueeze(i, max_node_num, max_node_num, max_dist) for i in edge_inputs]
#     )
#     attn_bias = torch.cat(
#         [pad_attn_bias_unsqueeze(i, max_node_num + 1) for i in attn_biases]
#     )
#     attn_edge_type = torch.cat(
#         [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types]
#     )
#     spatial_pos = torch.cat(
#         [pad_spatial_pos_unsqueeze(i, max_node_num) for i in spatial_poses]
#     )
#     in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num) for i in in_degrees])

#     return dict(
#         idx=torch.LongTensor(idxs),
#         attn_bias=attn_bias,
#         attn_edge_type=attn_edge_type,
#         spatial_pos=spatial_pos,
#         in_degree=in_degree,
#         out_degree=in_degree,  # for undirected graph
#         x=x,
#         edge_input=edge_input,
#         y=y,
#     )


######### my functions ##################

class Batch_graphormer():
    def __init__(self, idx, attn_bias, attn_edge_type, spatial_pos, in_degree, out_degree, x, edge_input, y):
        super(Batch_graphormer, self).__init__()
        self.idx = idx
        self.in_degree, self.out_degree = in_degree, out_degree
        self.x, self.y = x, y
        self.attn_bias, self.attn_edge_type, self.spatial_pos = attn_bias, attn_edge_type, spatial_pos
        self.edge_input = edge_input

    def to(self, device):
        self.idx = self.idx.to(device)
        self.in_degree, self.out_degree = self.in_degree.to(
            device), self.out_degree.to(device)
        self.x, self.y = self.x.to(device), self.y.to(device)
        self.attn_bias, self.attn_edge_type, self.spatial_pos = self.attn_bias.to(
            device), self.attn_edge_type.to(device), self.spatial_pos.to(device)
        self.edge_input = self.edge_input.to(device)
        return self

    def __len__(self):
        return self.in_degree.size(0)
    
class Batch_baseline():
    def __init__(self, idx, x, edge_index, y):
        super(Batch_baseline, self).__init__()
        self.idx = idx
        self.x, self.y = x, y
        self.edge_index = edge_index.T

    def to(self, device):
        self.idx = self.idx.to(device)
        self.x, self.y = self.x.float().to(device), self.y.to(device)
        self.edge_index = self.edge_index.to(device)
        return self
    
    

def collator_graphormer(items, max_node=300, multi_hop_max_dist=1, spatial_pos_max=20):
    max_node = 300000
    multi_hop_max_dist = 1
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    

    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y) for item in items]

    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys = zip(
        *items)

    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)

    y = torch.cat(ys)

    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])

    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                           for i in out_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        edge_input=edge_input,
        y=y
    )

def collator_graphormer_link(items, max_node=300, multi_hop_max_dist=1, spatial_pos_max=20):
    max_node = 300000
    multi_hop_max_dist = 1
    items = [
        item for item in items if item is not None and item.x.size(0) <= max_node]
    
    # print('nodes_items 1', nodes_items)
    # print(num_nodesefdfdfd)
    items = [(item.idx, item.attn_bias, item.attn_edge_type, item.spatial_pos, item.in_degree,
              item.out_degree, item.x, item.edge_input[:, :, :multi_hop_max_dist, :], item.y, item.edge_label_pos, item.edge_label_neg, item.edge_label_index_pos, item.edge_label_index_neg) for item in items]
#     print('items 2', items)
    idxs, attn_biases, attn_edge_types, spatial_poses, in_degrees, out_degrees, xs, edge_inputs, ys, edge_label_poses, edge_label_negos, edge_label_index_poses, edge_label_index_negos = zip(
        *items)
    # print('num_nodeses', num_nodeses[0])
    # print(num_nodesefdfdfd)
    for idx, _ in enumerate(attn_biases):
        attn_biases[idx][1:, 1:][spatial_poses[idx] >= spatial_pos_max] = float('-inf')
    max_node_num = max(i.size(0) for i in xs)
    max_dist = max(i.size(-2) for i in edge_inputs)

    y = torch.cat(ys)

    x = torch.cat([pad_2d_unsqueeze(i, max_node_num) for i in xs])
    
    edge_label_pos = torch.cat(edge_label_poses)
    edge_label_neg = torch.cat(edge_label_negos)
    edge_label_index_pos = torch.cat(edge_label_index_poses)
    edge_label_index_neg = torch.cat(edge_label_index_negos)
    
    edge_input = torch.cat([pad_3d_unsqueeze(
        i, max_node_num, max_node_num, max_dist) for i in edge_inputs])
    attn_bias = torch.cat([pad_attn_bias_unsqueeze(
        i, max_node_num + 1) for i in attn_biases])
    attn_edge_type = torch.cat(
        [pad_edge_type_unsqueeze(i, max_node_num) for i in attn_edge_types])
    spatial_pos = torch.cat([pad_spatial_pos_unsqueeze(i, max_node_num)
                        for i in spatial_poses])
    in_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                          for i in in_degrees])
    out_degree = torch.cat([pad_1d_unsqueeze(i, max_node_num)
                           for i in out_degrees])

    return dict(
        idx=torch.LongTensor(idxs),
        attn_bias=attn_bias,
        attn_edge_type=attn_edge_type,
        spatial_pos=spatial_pos,
        in_degree=in_degree,
        out_degree=out_degree,
        x=x,
        edge_input=edge_input,
        y=y,
        edge_label_pos = edge_label_pos,
        edge_label_neg = edge_label_neg, 
        edge_label_index_pos = edge_label_index_pos,
        edge_label_index_neg = edge_label_index_neg
    )


def collator_baseline(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20):
    items = [item for item in items if item is not None]
    items = [(item.idx, item.edge_index.T, item.x,  item.y) for item in items]
    idxs, edge_indexes, xs, ys = zip(*items)
    
    y = torch.cat(ys)
    x = torch.cat(xs)
    edge_index = torch.cat(edge_indexes)

    return dict(
        idx=torch.LongTensor(idxs),
        x=x,
        edge_index = edge_index,
        y=y
    )


def collator(items, max_node=512, multi_hop_max_dist=20, spatial_pos_max=20, mode = 'graphormer'):
    if mode == 'graphormer':
        return collator_graphormer(items, max_node, multi_hop_max_dist, spatial_pos_max)
    elif mode == 'baseline':
        return collator_baseline(items, max_node, multi_hop_max_dist, spatial_pos_max)
    elif mode == 'graphormer_link':
        return collator_graphormer_link(items, max_node, multi_hop_max_dist, spatial_pos_max)