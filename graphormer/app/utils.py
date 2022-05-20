import numpy as np
import os.path as osp
import torch
import torch.nn.functional as F
from data_class import prepare_raw_dataset_edge, prepare_raw_dataset_node, full_geo_Abakan
from torch_geometric.data import Data
import pandas as pd
import numpy as np

from evaluate_points import prepare_dataset, prepare_args, prepare_points, prepare_eval_model, prepare_eval_iterator, predict_time, prepare_task, convert_to_torch
from pathlib import Path
import os
import sys
from torch import Tensor
# BASE = Path(os.path.realpath('')).parent
# GLOBAL_ROOT = str(BASE / 'app' / 'graphormer_repo' / 'graphormer')
BASE = str(Path(os.path.realpath('')).parent) + '/app'

from fairseq.dataclass.utils import convert_namespace_to_omegaconf

def haversine_np(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)

    All args must be of equal length.    

    """
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    m = 6367 * c * 1000
    return m

def get_perp( X1, Y1, X2, Y2, X3, Y3):
    """************************************************************************************************ 
    Purpose - X1,Y1,X2,Y2 = Two points representing the ends of the line segment
              X3,Y3 = The offset point 
    'Returns - X4,Y4 = Returns the Point on the line perpendicular to the offset or None if no such
                        point exists
    '************************************************************************************************ """
    XX = X2 - X1 
    YY = Y2 - Y1 
    ShortestLength = ((XX * (X3 - X1)) + (YY * (Y3 - Y1))) / ((XX * XX) + (YY * YY))
    print(ShortestLength)
    X4 = X1 + XX * ShortestLength 
    Y4 = Y1 + YY * ShortestLength
    if X4 < X2 and X4 > X1 and Y4 < Y2 and Y4 > Y1:
        return X4,Y4
    return None


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1 if edge_index.numel() > 0 else 0
    else:
        return max(edge_index.size(0), edge_index.size(1))
    
def k_hop_subgraph(node_idx, num_hops, edge_index, relabel_nodes=False,
                   num_nodes=None, flow='source_to_target'):

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index
    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    node_mask.fill_(False)
    node_mask[subset] = True
    edge_mask = node_mask[row] & node_mask[col]

    edge_index = edge_index[:, edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, edge_mask

def make_subgraph(patch, data):
    index, adj, mapping, mask = k_hop_subgraph(
        patch,
        2,
        data.edge_index,
        num_nodes=None,
        relabel_nodes = True
    )
    index = index.numpy()
    feature = data.x[index]
    label = data.y[index]
    edge_attr = data.edge_attr[mask]
    new_data = Data(x=feature, edge_index=adj, y=label, edge_attr = edge_attr)
    return new_data

def point_predict(edge, all_graph, cfg, model_abakan, task_abakan):
    test_edge = edge
    cur_graph = make_subgraph(test_edge, all_graph)
    predict_torch_dataset = [cur_graph]
    predict_torch_dataset[0].y = torch.tensor(1)
    iterator = prepare_eval_iterator(cfg, predict_torch_dataset, 'abakan', task_abakan)
    
    if len(cur_graph.edge_index[0]) < 60:
        time = predict_time(model_abakan, iterator)
        time = time / cur_graph.edge_index.size()[1]
    else:
        time = 6.123
    return time

def get_weights(raw_edges, all_graph, cfg, model, task):
    node_time_dict = dict()
    for i in range(0, all_graph.num_nodes):
        if i%1000 == 0:
            print(i)
        try:
            time = point_predict([i], all_graph, cfg, model, task)
            node_time_dict[i] = time
        except:
            node_time_dict[i] = 6.123
    edges_source_list = [x[0] for x in raw_edges]
    weights_list = [node_time_dict[i] for i in edges_source_list]
    return np.array(weights_list)
