import torch
import numpy as np
from fairseq import checkpoint_utils, utils, options, tasks
from fairseq.logging import progress_bar
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
import ogb
import sys
import os
from pathlib import Path
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

from os import path
import logging
from data_class import single_geo_Omsk, GraphormerPYGDataset_predict, single_geo_Abakan, single_geo_Abakan_raw
import os.path as osp
from torch_geometric.data import Dataset
from functools import lru_cache
import torch_geometric.datasets
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
import pyximport
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from sklearn import preprocessing
pyximport.install(setup_args={'include_dirs': np.get_include()})
from torch_geometric.data import Data
import time
from torch_geometric.utils import add_self_loops, negative_sampling
import copy
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
)
import json
import pathlib
from pathlib import Path
BASE = Path(os.path.realpath(__file__)).parent
GLOBAL_ROOT = str(BASE / 'graphormer_repo' / 'graphormer')
BASE = str(Path(os.path.realpath('')).parent) + '/app'
sys.path.insert(2, (GLOBAL_ROOT))
from data.wrapper import preprocess_item

from pretrain import load_pretrained_model
from data.pyg_datasets.pyg_dataset import GraphormerPYGDataset
from data.dataset import (
    BatchedDataDataset,
    TargetDataset,
    GraphormerDataset)


def prepare_eval_iterator(cfg, predict_dataset, data_name, task):
    # print('prepare_eval_iterator start')
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    seed = 71
    GPYG = GraphormerPYGDataset_predict(predict_dataset, seed, None, predict_dataset, data_name)
    batched_data = BatchedDataDataset(GPYG)
    data_sizes = np.array([128] * len(batched_data))
    dataset_total = NestedDictionaryDataset(
            {
                "nsamples": NumSamplesDataset(),
                "net_input": {"batched_data": batched_data},
                "target": batched_data,
            },
        sizes=data_sizes,
        )
    # task = tasks.setup_task(cfg.task)
    batch_iterator = task.get_batch_iterator(
        dataset=dataset_total
    )
    itr = batch_iterator.next_epoch_itr(shuffle=False, set_dataset_epoch=False)
    progress = progress_bar.progress_bar(itr)
    # print('prepare_eval_iterator end')
    return progress

def prepare_task(args):
    # print('prep task')
    cfg = convert_namespace_to_omegaconf(args)
    task = tasks.setup_task(cfg.task)
    return task

def prepare_eval_model(args, model_state_link):
    model_state = torch.load(model_state_link)["model"]
    cfg = convert_namespace_to_omegaconf(args)
    np.random.seed(cfg.common.seed)
    utils.set_torch_seed(cfg.common.seed)
    seed = 71
    task = tasks.setup_task(cfg.task)
    model = task.build_model(cfg.model)
    model.load_state_dict(model_state, strict=True, model_cfg=cfg.model)
    model.to(torch.cuda.current_device())
    return model

def prepare_args(dataset_name):
    # print('prep argsss')
    parser_dict = dict()
    parser_dict['num-atoms'] = str(6656)
    parser_dict['dataset_name'] = dataset_name
    train_parser = options.get_training_parser()
    train_parser.add_argument(
            "--split",
            type=str,
        )
    train_parser.add_argument(
            "--metric",
            type=str,
        )
    train_parser.add_argument(
            "--dataset_name",
            type=str,
        )
    train_args = options.parse_args_and_arch(
        train_parser,
        [
            '--user-dir' , GLOBAL_ROOT,
            '--num-workers' , '10', 
            '--ddp-backend' , 'legacy_ddp', 
            '--dataset_name' , parser_dict['dataset_name'], 
            '--dataset-source' , 'pyg', 
            '--num-atoms' , parser_dict['num-atoms'], 
            '--task' , 'graph_prediction', 
            '--criterion' , 'l1_loss', 
            '--arch' , 'graphormer_slim',
            '--num-classes' , '1', 
            '--batch-size' , '1', 
            '--save-dir' ,  BASE + '/models/'+ dataset_name,
            '--split' , 'valid', 
            '--metric' , 'rmse',
            '--mode', 'predict'
        ]
    )
    
    args = train_args
    return args

def predict_time(model, progress):
    # print('predict_time start')
    y_pred = []
    with torch.no_grad():
        model.eval()
        for i, sample in enumerate(progress):
            # print('sample')
            sample = utils.move_to_cuda(sample)
            # print('sample')
            y = model(**sample["net_input"])[:, 0, :].reshape(-1)
            y_pred.extend(y.detach().cpu())
            torch.cuda.empty_cache()
            
    # save predictions
    y_pred = torch.Tensor(y_pred) 
    # print('y_pred:', y_pred)
    # print('predict_time end')
    return y_pred   

def prepare_dataset(dataset_link):
    data = pd.read_pickle(dataset_link).dropna()
    # data['edge_coord_start'] = data['edge_coord_start'].apply(lambda x: json.loads(x))
    # data['edge_coord_end'] = data['edge_coord_end'].apply(lambda x: json.loads(x))
    data['edge_coord_start_N'] = data['edge_coord_start'].apply(lambda x: x[0][0])
    data['edge_coord_start_E'] = data['edge_coord_start'].apply(lambda x: x[0][1])
    data['edge_coord_end_N'] = data['edge_coord_end'].apply(lambda x: x[0][0])
    data['edge_coord_end_E'] = data['edge_coord_end'].apply(lambda x: x[0][1])
    return data

def prepare_points(data, pt_start_N, pt_start_E, pt_end_N, pt_end_E, data_name = ''):
    # print('prepare_points start')
    # point_start = pt_start
    # point_end = pt_end

    data['point_start_N'] = pt_start_N
    data['point_start_E'] = pt_start_E
    data['point_end_N'] = pt_end_N
    data['point_end_E'] = pt_end_E
    
    print('apply start')
    # data['dist_start'] = data.apply(lambda x: (x['edge_coord_start'][0][0] - x['point_start_N'])**2 + (x['edge_coord_start'][0][1] - x['point_start_E'])**2, axis = 1)
    # data['dist_end'] = data.apply(lambda x: (x['edge_coord_end'][0][0] - x['point_end_N'])**2 + (x['edge_coord_end'][0][1] - x['point_end_E'])**2, axis = 1)
    # data['dist_mean'] = (data['dist_start'] + data['dist_end'])/2
    
    
    data['dist_start'] = (data['edge_coord_start_N'] - data['point_start_N'])**2 + (data['edge_coord_start_E'] - data['point_start_E'])**2
    data['dist_end'] = (data['edge_coord_end_N'] - data['point_end_N'])**2 + (data['edge_coord_end_E'] - data['point_end_E'])**2
    data['dist_mean'] = (data['dist_start'] + data['dist_end'])/2
    # print('apply start')
    
    predict_table = data.sort_values(by = ['dist_mean']).reset_index(drop = True)[:1]
    # print('prepare_points end')
    return predict_table

def convert_to_torch(predict_table, data_name):
    # print('geo start')
    if data_name == 'abakan':
        # dataset = single_geo_Abakan(predict_table)
        dataset = single_geo_Abakan_raw(predict_table)
    if data_name == 'omsk':
        dataset = single_geo_Omsk(predict_table)
    dataset = dataset.process()
    # print('geo end')
    # print('prepare_points end')
    return dataset
    
    
def graphormer_predict(pt_start, pt_end, dataset_name, data, model_state):
    dataset_name = 'abakan'
    dataset_abakan_link = BASE + '/datasets/' + dataset_name + '/raw/final.csv'
    model_abakan_link = BASE + '/models/'+ dataset_name + '/checkpoint_last.pt'
    data_abakan = prepare_dataset(dataset_abakan_link)
    data_abakan = prepare_points(data_abakan, pt_start, pt_end)

    model_state_abakan = torch.load(model_abakan_link)["model"]
    predict_table = data.sort_values(by = ['dist_mean']).reset_index(drop = True)[:1]

    dataset = single_geo_Abakan(predict_table)
    dataset = dataset.process()
    predicted_time = predict_time(dataset_name, dataset, model_state)

    return [predict_table['edges_geo'], predicted_time]

def check_town(points): 
    min_N_abk = 90.91763111635001
    max_N_abk = 91.88558398090001
    min_E_abk = 52.84332097535
    max_E_abk = 53.9852115715

    min_N_omsk = 72.8949037781
    max_N_omsk = 73.75839039050001
    min_E_omsk = 54.78700068105
    max_E_omsk = 55.39520542775

    if min(points.start_lat, points.end_lat) >= min_N_abk and max(points.start_lat, points.end_lat) <= max_N_abk and min(points.start_lon, points.end_lon) >= min_E_abk and max(points.start_lon, points.end_lon) <= max_E_abk:
        return 'abakan'
    elif min(points.start_lat, points.end_lat) >= min_N_omsk and max(points.start_lat, points.end_lat) <= max_N_omsk and min(points.start_lon, points.end_lon) >= min_E_omsk and max(points.start_lon, points.end_lon) <= max_E_omsk:
        return 'omsk'
    else:
        return 0
