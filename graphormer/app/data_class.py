import sys
import os
import torch
import numpy as np
import torch_geometric.datasets
import pyximport
from torch_geometric.data import InMemoryDataset, download_url
import pandas as pd
from sklearn import preprocessing
pyximport.install(setup_args={'include_dirs': np.get_include()})
import os.path as osp
from torch_geometric.data import Data
import time
from torch_geometric.utils import add_self_loops, negative_sampling
from torch_geometric.data import Dataset
from functools import lru_cache
import copy
from fairseq.data import (
    NestedDictionaryDataset,
    NumSamplesDataset,
)

import json
import pathlib
from pathlib import Path

BASE = Path(os.path.realpath(__file__)).parent
GLOBAL_ROOT = str(BASE  / 'graphormer_repo' / 'graphormer')
sys.path.insert(1, (GLOBAL_ROOT))
from data.wrapper import preprocess_item


import datetime

def find_part(hour):
    if hour < 11:
        part = 1
    elif (hour > 11) & (hour < 20):
        part = 2
    else:
        part = 3
    return part


def prepare_raw_dataset_edge(dataset_name):
    if dataset_name == 'abakan':
        raw_data = pd.read_csv('datasets/abakan/raw/abakan_full_routes_final_weather_L_NaN_filtered_FIXED.csv')
        all_roads_graph = list(pd.read_pickle('datasets/abakan/raw/all_roads_graph.pickle').to_networkx().edges())
        all_nodes = pd.read_pickle('datasets/abakan/raw/clear_nodes.pkl')
        init = pd.read_csv('datasets/abakan/raw/graph_abakan_init.csv')
    elif dataset_name == 'omsk':
        raw_data = pd.read_csv('datasets/omsk/raw/omsk_full_routes_final_weather_L_NaN_filtered_FIXED.csv')
        all_roads_graph = list(pd.read_pickle('datasets/omsk/raw/all_roads_graph.pickle').to_networkx().edges())
        # all_nodes = pd.read_pickle('datasets/omsk/raw/clear_nodes.pkl')
        init = pd.read_csv('datasets/omsk/raw/graph_omsk_init.csv')
    all_roads_dataset = pd.DataFrame()
    all_edge_list = [list((all_roads_graph)[i]) for i in range(0,len( (all_roads_graph)))]

    all_roads_dataset['edge_id']= range(0,len(init['edge_id'].unique()))
    all_roads_dataset['speed'] = ' 1'
    all_roads_dataset['length'] = ' 1'
    all_roads_dataset[' start_point_part'] = init['quartal_id'] / len(init['quartal_id'].unique())
    all_roads_dataset['finish_point_part'] = init['quartal_id'] / len(init['quartal_id'].unique())
    
    all_roads_dataset_edges = pd.DataFrame()
    all_roads_dataset_edges['source'] = [x[0] for x in all_edge_list]
    all_roads_dataset_edges['target'] = [x[1] for x in all_edge_list]
    # all_roads_dataset_edges = all_roads_dataset_edges.drop_duplicates().reset_index(drop = True)
    
    trip_part = all_roads_dataset[['edge_id', 'speed', 'length', ' start_point_part', 'finish_point_part']].copy()
    source_merge = pd.merge(all_roads_dataset_edges, trip_part.rename(columns = {'edge_id':'source'}), on = ['source'], how = 'left')
    target_merge = pd.merge(all_roads_dataset_edges, trip_part.rename(columns = {'edge_id':'target'}), on = ['target'], how = 'left')

    total_table = pd.DataFrame()
    total_table['speed'] = (source_merge['speed'].apply(lambda x: [x]) + target_merge['speed'].apply(lambda x: [x]))
    total_table['length'] = (source_merge['length'].apply(lambda x: [x]) + target_merge['length'].apply(lambda x: [x]))
    total_table['edges'] = (source_merge['source'].apply(lambda x: [x]) + target_merge['target'].apply(lambda x: [x]))
    total_table[' start_point_part'] = source_merge[' start_point_part']
    total_table['finish_point_part'] = target_merge['finish_point_part']

    total_table['week_period'] = datetime.datetime.now().hour
    total_table['hour'] = datetime.datetime.now().weekday()
    total_table['day_period'] = total_table['hour'].apply(lambda x: find_part(x))

    total_table['RTA'] = 1

    total_table['clouds'] = 1
    total_table['snow'] = 0
    total_table['temperature'] = 10
    total_table['wind_dir'] = 180
    total_table['wind_speed'] = 3
    total_table['pressure'] = 747
    
    total_table['source'] = source_merge['source']
    total_table['target'] = source_merge['target']
    
    # total_table = total_table.drop_duplicates().reset_index(drop = True)
    return total_table

def prepare_raw_dataset_node(dataset_name):
    if dataset_name == 'abakan':
        raw_data = pd.read_csv('datasets/abakan/raw/abakan_full_routes_final_weather_L_NaN_filtered_FIXED.csv')
        all_roads_graph = list(pd.read_pickle('datasets/abakan/raw/all_roads_graph.pickle').to_networkx().edges())
        all_nodes = pd.read_pickle('datasets/abakan/raw/clear_nodes.pkl')
        init = pd.read_csv('datasets/abakan/raw/graph_abakan_init.csv')
    elif dataset_name == 'omsk':
        raw_data = pd.read_csv('datasets/omsk/raw/omsk_full_routes_final_weather_L_NaN_filtered_FIXED.csv')
        all_roads_graph = list(pd.read_pickle('datasets/omsk/raw/all_roads_graph.pickle').to_networkx().edges())
        # all_nodes = pd.read_pickle('datasets/omsk/raw/clear_nodes.pkl')
        init = pd.read_csv('datasets/omsk/raw/graph_omsk_init.csv')
    all_roads_dataset = pd.DataFrame()
    all_edge_list = [list((all_roads_graph)[i]) for i in range(0,len( (all_roads_graph)))]

    all_roads_dataset['edge_id']= range(0,len(init['edge_id'].unique()))
    all_roads_dataset['speed'] = ' 1'
    all_roads_dataset['length'] = ' 1'
    all_roads_dataset[' start_point_part'] = init['quartal_id'] / len(init['quartal_id'].unique())
    all_roads_dataset['finish_point_part'] = init['quartal_id'] / len(init['quartal_id'].unique())
    
    all_roads_dataset['finish_point_part'] = all_roads_dataset['finish_point_part']

    all_roads_dataset['week_period'] = datetime.datetime.now().hour
    all_roads_dataset['hour'] = datetime.datetime.now().weekday()
    all_roads_dataset['day_period'] = all_roads_dataset['hour'].apply(lambda x: find_part(x))

    all_roads_dataset['RTA'] = 1

    all_roads_dataset['clouds'] = 1
    all_roads_dataset['snow'] = 0
    all_roads_dataset['temperature'] = 10
    all_roads_dataset['wind_dir'] = 180
    all_roads_dataset['wind_speed'] = 3
    all_roads_dataset['pressure'] = 747
    
    # all_roads_dataset['source'] = source_merge['source']
    # all_roads_dataset['target'] = source_merge['target']
    
    # total_table = total_table.drop_duplicates().reset_index(drop = True)
    return all_roads_dataset

    
class single_geo_Omsk(InMemoryDataset):
    def __init__(self, predict_data, transform=None, pre_transform=None, split = 'train'):
        self.data = predict_data

    def process(self):
        
        # Read data
        # print('start single')
        start_time = time.time()
        data = self.data
        # shape = int(data.shape[0]÷)
        shape = int(10)
        data = data[0:1].copy()
        
        data = data.drop(columns = ['Unnamed: 0'])
        data['hour'] = data['start_timestamp'].apply(lambda x: int(x[-10:-8]))
        # Graph 
        graph_columns_gran = ['edges', 'time', 'speed', 'length']
        edges = ['edges']
        target = ['time']
        node_features_gran = ['speed', 'length']

        edge_features_agg = [' start_point_part', 'finish_point_part', 'day_period', 'week_period', 'clouds', 'snow', 'temperature', 'wind_dir', 'wind_speed', 'pressure','hour']

        
        all_speed = []
        all_length = []
        for i in range(0,1):
            # print(i)
            data_row = data[i:i+1].reset_index(drop = True).copy()
            speed_list = [int(x) for x in (data_row['speed'].values[0].replace("'",'').split(','))]
            list_length = [int(x) for x in (data_row['length'].values[0].replace("'",'').split(','))]
            all_speed.append(speed_list)
            all_length.append(list_length)
            
        all_speed = [item for sublist in all_speed for item in sublist]
        all_length = [item for sublist in all_length for item in sublist]
    
        data_split_dict = dict()
        data_split_dict['all'] = np.arange(0, int(data.shape[0]))
        
        data_list = []
        for i in data_split_dict['all']:
            data_row = data.iloc[[i],].reset_index(drop = True).copy()

            speed_list = [int(x) for x in (data_row['speed'].values[0].replace("'",'').split(','))]
            list_length = [int(x) for x in (data_row['length'].values[0].replace("'",'').split(','))]


            data_row_gran = pd.DataFrame()
            data_row_gran['source'] = data_row['source']
            data_row_gran['target'] = data_row['target']
            data_row_gran['speed'] = speed_list
            data_row_gran['length'] = list_length


            target_val = data_row['RTA'].values[0]


            data_row_gran['speed'] = data_row_gran['speed']/np.mean(speed_list)
            data_row_gran['length'] = data_row_gran['length']/np.mean(list_length)

            for col in edge_features_agg:
                data_row_gran[col] = data_row[col].values[0]

            total_nodes_list = list(set(list(data_row_gran.source.values)))
            le = preprocessing.LabelEncoder()
            le.fit(total_nodes_list)

            data_row_gran['source'] = le.transform(data_row_gran.source.values)
            data_row_gran['target'] = le.transform(data_row_gran.target.values)

            total_nodes_list = list(set(list(data_row_gran.source.values)))

            edge_index = torch.tensor(torch.from_numpy(data_row_gran[['source','target']].values.T),dtype = torch.long)


            # Define tensor of nodes features
            x = torch.tensor(torch.from_numpy(data_row_gran[['speed','length'] + edge_features_agg].values),dtype = torch.long)


            # Define tensor of edge features
            edge_num_feach = 1
            edge_attr = torch.from_numpy(np.ones(shape = ((edge_index.size()[1]), edge_num_feach)))
            edge_attr = torch.tensor(edge_attr,dtype = torch.long)

            # Define tensor of targets
            y = torch.tensor(target_val,dtype = torch.long)


            data_graph = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y=y)
            data_list.append(data_graph)
            # print('end single')
            return data_list
        
        

class single_geo_Abakan_raw():
    def __init__(self, predict_data, transform=None, pre_transform=None, split = 'train'):
        self.data = predict_data

    def process(self):
        
        # Read data
        # print('start single')
        start_time = time.time()
        data = self.data
        # shape = int(data.shape[0]÷)
        shape = int(10)
        data = data[0:1].copy()
        
        data = data.drop(columns = ['Unnamed: 0'])
        data['hour'] = data['start_timestamp'].apply(lambda x: int(x[-10:-8]))
        # Graph 
        graph_columns_gran = ['edges', 'time', 'speed', 'length']
        edges = ['edges']
        target = ['time']
        node_features_gran = ['speed', 'length']

        edge_features_agg = [' start_point_part', 'finish_point_part', 'day_period', 'week_period', 'clouds', 'snow', 'temperature', 'wind_dir', 'wind_speed', 'pressure','hour']

        
        all_speed = []
        all_length = []
        for i in range(0,1):
            # print(i)
            data_row = data[i:i+1].reset_index(drop = True).copy()
            speed_list = [int(x) for x in (data_row['speed'].values[0].replace("'",'').split(','))]
            list_length = [int(x) for x in (data_row['length'].values[0].replace("'",'').split(','))]
            all_speed.append(speed_list)
            all_length.append(list_length)
            
        all_speed = [item for sublist in all_speed for item in sublist]
        all_length = [item for sublist in all_length for item in sublist]
    
        data_split_dict = dict()
        data_split_dict['all'] = np.arange(0, int(data.shape[0]))
        
        data_list = []
        for i in data_split_dict['all']:
            data_row = data.iloc[[i],].reset_index(drop = True).copy()

            edge_list = [int(x) for x in (data_row['edges'].values[0].replace("'",'').split(','))]
            speed_list = [int(x) for x in (data_row['speed'].values[0].replace("'",'').split(','))]
            list_length = [int(x) for x in (data_row['length'].values[0].replace("'",'').split(','))]

            source = edge_list.copy()
            target = edge_list[1:].copy() + [edge_list[0]].copy()

            data_row_gran = pd.DataFrame()
            data_row_gran['source'] = source
            data_row_gran['target'] = target
            data_row_gran['speed'] = speed_list
            data_row_gran['length'] = list_length


            target_val = data_row['RTA'].values[0]


            data_row_gran['speed'] = data_row_gran['speed']/np.mean(speed_list)
            data_row_gran['length'] = data_row_gran['length']/np.mean(list_length)

            for col in edge_features_agg:
                data_row_gran[col] = data_row[col].values[0]

            total_nodes_list = list(set(list(data_row_gran.source.values)))
            le = preprocessing.LabelEncoder()
            le.fit(total_nodes_list)

            data_row_gran['source'] = le.transform(data_row_gran.source.values)
            data_row_gran['target'] = le.transform(data_row_gran.target.values)

            total_nodes_list = list(set(list(data_row_gran.source.values)))

            edge_index = torch.tensor(torch.from_numpy(data_row_gran[['source','target']].values.T),dtype = torch.long)


            # Define tensor of nodes features
            x = torch.tensor(torch.from_numpy(data_row_gran[['speed','length'] + edge_features_agg].values),dtype = torch.long)


            # Define tensor of edge features
            edge_num_feach = 1
            edge_attr = torch.from_numpy(np.ones(shape = ((edge_index.size()[1]), edge_num_feach)))
            edge_attr = torch.tensor(edge_attr,dtype = torch.long)

            # Define tensor of targets
            y = torch.tensor(target_val,dtype = torch.long)


            data_graph = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y=y)
            data_list.append(data_graph)
            # print('end single')
            return data_list
        
        

class GraphormerPYGDataset_predict(Dataset):
    def __init__(
        self,
        dataset: Dataset,
        seed: int = 0,
        predict_idx=None,
        predict_set=None,
        name = None
    ):
        self.name = name
        self.dataset = dataset
        if self.dataset is not None:
            self.num_data = len(self.dataset)
        self.seed = seed

        self.num_data = len(predict_set) 
        self.predict_idx = predict_idx
        self.predict_data = self.create_subset(predict_set)
        self.__indices__ = None

    def index_select(self, idx):
        dataset = copy.copy(self)
        dataset.dataset = self.dataset.index_select(idx)
        if isinstance(idx, torch.Tensor):
            dataset.num_data = idx.size(0)
        else:
            dataset.num_data = idx.shape[0]
        dataset.__indices__ = idx
        dataset.predict_data = None
        dataset.predict_idx = None
        return dataset

    def create_subset(self, subset):
        dataset = copy.copy(self)
        dataset.dataset = subset
        dataset.num_data = len(subset)
        dataset.__indices__ = None
        dataset.predict_data = None
        dataset.predict_idx = None
        return dataset

    @lru_cache(maxsize=16)
    def __getitem__(self, idx):
        if isinstance(idx, int):
            # print('idx', idx)
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data



    
    
class single_geo_Abakan():
    def __init__(self, predict_data, transform=None, pre_transform=None, split = 'train'):
        self.data = predict_data

    def process(self):
        
        # Read data
        data = self.data
                
        # Graph 
        graph_columns_gran = ['edges', 'time', 'speed', 'length']
        edges = ['edges']
        target = ['time']
        node_features_gran = ['speed', 'length']

        edge_features_agg = [' start_point_part', 'finish_point_part', 'day_period', 'week_period', 'clouds', 'snow', 'temperature', 'wind_dir', 'wind_speed', 'pressure','hour']

    
        data_split_dict = dict()
        data_split_dict['all'] = np.arange(0, int(data.shape[0]))
        
        data_list = []
        data_row_gran = pd.DataFrame()
        data_row_gran['source'] = [0,1]
        data_row_gran['target'] = [1,0]
        data_row_gran['speed'] = [1,1]
        data_row_gran['length'] = [1,1]
        target_val = 1
        edge_index = torch.tensor(torch.from_numpy(data_row_gran[['source','target']].values.T),dtype = torch.long)
        
        # Define tensor of edge features
        edge_num_feach = 1
        edge_attr = torch.from_numpy(np.ones(shape = ((edge_index.size()[1]), edge_num_feach)))
        edge_attr = torch.tensor(edge_attr,dtype = torch.long)
        for col in edge_features_agg:
            data_row_gran[col] = data.iloc[[1],].reset_index(drop = True)[col].values[0]
            
        # Define tensor of targets
        y = torch.tensor(target_val,dtype = torch.long)
            
        for i in data_split_dict['all']:
            if (i % 10000 ) == 0 :
                print(i)
            data_row = data.iloc[[i],].reset_index(drop = True).copy()
            for col in [' start_point_part', 'finish_point_part']:
                data_row_gran[col] = data_row[col].values[0]
            # Define tensor of nodes features
            x = torch.tensor(torch.from_numpy(data_row_gran[['speed','length'] + edge_features_agg].values),dtype = torch.long)

            data_graph = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y=y)
            data_list.append(data_graph)
            # print('end single')
        return data_list
    
    
    
class single_geo_Omsk():
    def __init__(self, predict_data, transform=None, pre_transform=None, split = 'train'):
        self.data = predict_data

    def process(self):
        
        # Read data
        data = self.data
                
        # Graph 
        graph_columns_gran = ['edges', 'time', 'speed', 'length']
        edges = ['edges']
        target = ['time']
        node_features_gran = ['speed', 'length']

        edge_features_agg = [' start_point_part', 'finish_point_part', 'day_period', 'week_period', 'clouds', 'snow', 'temperature', 'wind_dir', 'wind_speed', 'pressure','hour']

    
        data_split_dict = dict()
        data_split_dict['all'] = np.arange(0, int(data.shape[0]))
        
        data_list = []
        data_row_gran = pd.DataFrame()
        data_row_gran['source'] = [0,1]
        data_row_gran['target'] = [1,0]
        data_row_gran['speed'] = [1,1]
        data_row_gran['length'] = [1,1]
        target_val = 1
        edge_index = torch.tensor(torch.from_numpy(data_row_gran[['source','target']].values.T),dtype = torch.long)
        
        # Define tensor of edge features
        edge_num_feach = 1
        edge_attr = torch.from_numpy(np.ones(shape = ((edge_index.size()[1]), edge_num_feach)))
        edge_attr = torch.tensor(edge_attr,dtype = torch.long)
        for col in edge_features_agg:
            data_row_gran[col] = data.iloc[[1],].reset_index(drop = True)[col].values[0]
            
        # Define tensor of targets
        y = torch.tensor(target_val,dtype = torch.long)
            
        for i in data_split_dict['all']:
            if (i % 10000 ) == 0 :
                print(i)
            data_row = data.iloc[[i],].reset_index(drop = True).copy()
            for col in [' start_point_part', 'finish_point_part']:
                data_row_gran[col] = data_row[col].values[0]
            # Define tensor of nodes features
            x = torch.tensor(torch.from_numpy(data_row_gran[['speed','length'] + edge_features_agg].values),dtype = torch.long)

            data_graph = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y=y)
            data_list.append(data_graph)
            # print('end single')
        return data_list
    
    
    
class full_geo_Abakan():
    def __init__(self, prepared_dataset_edge, prepared_dataset_node, transform=None, pre_transform=None, split = 'train'):
        self.data_edge = prepared_dataset_edge
        self.data_node = prepared_dataset_node
        
    def process(self):
        
        # Read data
        data = self.data_node.copy()
        prepared_dataset_edge = self.data_edge.copy()
        
        # Graph 
        graph_columns_gran = ['edges', 'time', 'speed', 'length']
        edges = ['edges']
        target = ['time']
        node_features_gran = ['speed', 'length']

        edge_features_agg = [' start_point_part', 'finish_point_part', 'day_period', 'week_period', 'clouds', 'snow', 'temperature', 'wind_dir', 'wind_speed', 'pressure','hour']


        data_split_dict = dict()
        data_split_dict['all'] = np.arange(0, int(data.shape[0]))

        data_list = []
        data_row_gran = pd.DataFrame()
        data_row_gran['speed'] = data['speed'].apply(lambda x: int(x))
        data_row_gran['length'] = data['length'].apply(lambda x: int(x))
        target_val = torch.ones(data.shape[0])
        edge_index = torch.tensor(torch.from_numpy(prepared_dataset_edge[['source','target']].values.T),dtype = torch.long)

        # Define tensor of edge features
        edge_num_feach = 1
        edge_attr = torch.from_numpy(np.ones(shape = ((edge_index.size()[1]), edge_num_feach)))
        edge_attr = torch.tensor(edge_attr,dtype = torch.long)
        for col in edge_features_agg:
            data_row_gran[col] = data[col]


        # Define tensor of targets
        y = torch.tensor(target_val,dtype = torch.long)

        # for col in [' start_point_part', 'finish_point_part']:
        #         data_row_gran[col] = data_row[col].values[0]
            # Define tensor of nodes features
        x = torch.tensor(torch.from_numpy(data_row_gran[['speed','length'] + edge_features_agg].values),dtype = torch.long)

        data_graph = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y=y)
        return data_graph