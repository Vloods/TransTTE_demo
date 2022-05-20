import sys
import os

import torch
import numpy as np
import torch_geometric.datasets
from ogb.graphproppred import PygGraphPropPredDataset
from ogb.lsc.pcqm4m_pyg import PygPCQM4MDataset
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
sys.path.insert(2, '/home/jovyan/graphormer_v2/graphormer')
from data.wrapper import preprocess_item

class geo_Omsk(InMemoryDataset):
    
    
    def __init__(self, root, transform=None, pre_transform=None, split = 'train'):
        super(geo_Omsk, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_dir + '/' + f'{split}.pt')
        # self.data = torch.load(self.processed_dir + '/' + f'{split}.pt')
#         self.raw_dir = '/home/jovyan/'
        
    @property
    def raw_dir(self) -> str:
        return '/home/jovyan/tte_data/'
    
    
    @property
    def raw_file_names(self):
        return ['omsk_full_routes_final_weather_L_NaN_filtered_FIXED.csv']

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt', 'val.pt']
    
    @property
    def processed_dir(self):
        return osp.join(self.root)

    def download(self):
        path = download_url(self.url, self.raw_dir)
        print(self.processed_paths[0])
    
    def my_load_dataset(self):
        return [self.data, self.slices]
    
    def process(self):
        
        # Read data
        start_time = time.time()
        data = pd.read_csv(osp.join('/home/jovyan/tte_data/', 'omsk_full_routes_final_weather_L_NaN_filtered_FIXED.csv'))
        data = data[data['rebuildCount']<=1].reset_index(drop = True).copy()
        shape = int(data.shape[0])
        data = data[0:shape].copy()
        
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
        for i in range(0,shape):
            data_row = data[i:i+1].reset_index(drop = True).copy()
            speed_list = [int(x) for x in (data_row['speed'].values[0].replace("'",'').split(','))]
            list_length = [int(x) for x in (data_row['length'].values[0].replace("'",'').split(','))]
            all_speed.append(speed_list)
            all_length.append(list_length)
            
        all_speed = [item for sublist in all_speed for item in sublist]
        all_length = [item for sublist in all_length for item in sublist]
    
        train_size = 0.8
        test_size = 0.1
        val_size = 0.1

        data_split_dict = dict()
        data_split_dict['train'] = np.arange(0, int(data.shape[0]*train_size))
        data_split_dict['test'] = np.arange(int(data.shape[0]*train_size), int(data.shape[0]*(train_size+test_size)))
        data_split_dict['val'] = np.arange(int(data.shape[0]*(train_size + test_size)),int((data.shape[0]*(train_size+test_size + val_size))))
        
        for split in data_split_dict.keys():
            data_list = []
            for i in data_split_dict[split]:
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

                # Define tensor of edge features
                edge_num_feach = 1
                edge_attr = torch.from_numpy(np.ones(shape = ((edge_index.size()[1]), edge_num_feach)))
                edge_attr = torch.tensor(edge_attr,dtype = torch.long)

                # Define tensor of targets
                y = torch.tensor(target_val,dtype = torch.long)


                data_graph = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y=y)
                data_graph = preprocess_item(data_graph)
                data_list.append(data_graph)
            torch.save(data_list, osp.join(self.processed_dir, f'{split}.pt'))
    
    # def get(self, idx):
    #     data = torch.load(osp.join(self.processed_dir, f'{idx}.pt'))
    #     return data
    
    
class single_geo_Omsk(InMemoryDataset):
    
    
    def __init__(self, root, transform=None, pre_transform=None, split = 'train'):
        super(single_geo_Omsk, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_dir + '/predict_data.pt')
        # self.data = torch.load(self.processed_dir + '/' + f'{split}.pt')
#         self.raw_dir = '/home/jovyan/'
        
    @property
    def raw_dir(self) -> str:
        return '/home/jovyan/tte_data/'
    
    
    @property
    def raw_file_names(self):
        return ['omsk_full_routes_final_weather_L_NaN_filtered_FIXED.csv']

    @property
    def processed_file_names(self):
        return ['predict_data.pt']
    
    @property
    def processed_dir(self):
        return osp.join(self.root)

    def download(self):
        path = download_url(self.url, self.raw_dir)
        print(self.processed_paths[0])
    
    def my_load_dataset(self):
        return [self.data, self.slices]
    
    def process(self):
        
        # Read data
        start_time = time.time()
        data = pd.read_csv(osp.join('/home/jovyan/tte_data/', 'omsk_full_routes_final_weather_L_NaN_filtered_FIXED.csv'))
        data = data[data['rebuildCount']<=1].reset_index(drop = True).copy()
        # shape = int(data.shape[0]÷)
        shape = int(10)
        data = data[0:shape].copy()
        
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
        for i in range(0,shape):
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
        torch.save(self.collate(data_list), osp.join(self.processed_dir, 'predict_data.pt'))
        
        
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
            print('idx', idx)
            item = self.dataset[idx]
            item.idx = idx
            item.y = item.y.reshape(-1)
            return preprocess_item(item)
        else:
            raise TypeError("index to a GraphormerPYGDataset can only be an integer.")

    def __len__(self):
        return self.num_data

