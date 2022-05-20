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


import pathlib
link = pathlib.Path().resolve()
link = str(link).split('TransTTE')[0]
GLOBAL_ROOT = link + 'TransTTE'
        
class geo_Abakan(InMemoryDataset):
    
    
    def __init__(self, root, transform=None, pre_transform=None, split = 'train'):
        super(geo_Abakan, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_dir + '/' + f'{split}.pt')
#         self.raw_dir = '/home/jovyan/'
        
    @property
    def raw_dir(self) -> str:
        return GLOBAL_ROOT + '/datasets/abakan/raw'
    
    
    @property
    def raw_file_names(self):
        return ['abakan_full_routes_final_weather_L_NaN_filtered_FIXED.csv']

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt', 'val.pt']
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', 'data_abakan_1')

    def download(self):
        path = download_url(self.url, self.raw_dir)
        print(self.processed_paths[0])
    

    def process(self):
        
        # Read data
        print('start total')
        start_time = time.time()
        data = pd.read_csv(osp.join(GLOBAL_ROOT + '/datasets/abakan/raw', 'abakan_full_routes_final_weather_L_NaN_filtered_FIXED.csv'))
        data = data[data['rebuildCount']<=1].reset_index(drop = True).copy()
        shape = int(1*data.shape[0])
        data = data[0:shape].copy()
        
        data = data.drop(columns = ['Unnamed: 0'])
        data['hour'] = data['start_timestamp'].apply(lambda x: int(x[-10:-8]))
        # Graph 
        graph_columns_gran = ['edges', 'time', 'speed', 'length']
        edges = ['edges']
        target = ['time']
        node_features_gran = ['speed', 'length']
#         edge_features_agg = ['dist_to_b', 'dist_to_a', 'RTA', 'real_dist', 'pred_dist', 'rebuildCount', 'start_point_meters', 'finish_point_meters', ' start_point_part', 'finish_point_part', 'day_period', 'week_period','clouds', 'snow', 'temperature', 'wind_dir', 'wind_speed', 'pressure','hour']

        edge_features_agg = [' start_point_part', 'finish_point_part', 'day_period', 'week_period', 'clouds', 'snow', 'temperature', 'wind_dir', 'wind_speed', 'pressure','hour']
#         edge_features_agg = []
        
#         dict_mean = dict()
#         #### Normalization
#         for col in edge_features_agg:
#             dict_mean[col] = data[col].mean()
#             data[col] = data[col]/data[col].mean()
            
#         dict_mean['RTA'] = data['RTA'].mean()
#         data['RTA'] = data['RTA']/data['RTA'].mean()
        
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
        test_size = 0.15
        val_size = 0.05

        data_split_dict = dict()
        data_split_dict['train'] = np.arange(0, int(data.shape[0]*train_size))
        data_split_dict['test'] = np.arange(int(data.shape[0]*train_size), int(data.shape[0]*(train_size+test_size)))
        data_split_dict['val'] = np.arange(int(data.shape[0]*(train_size + test_size)),int((data.shape[0]*(train_size+test_size + val_size))))
        
        for split in data_split_dict.keys():
#             start_time = time.time()
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
                
                # dict_row_mean = dict_mean.copy()
                # dict_row_mean['speed'] = np.mean(speed_list)
                # dict_row_mean['length'] = np.mean(list_length)
                
                
                target_val = data_row['RTA'].values[0]


    #             data_row_gran['time'] = data_row_gran['time']/np.mean(all_time)
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
    #             edge_attr = torch.tensor(torch.from_numpy(data_row_gran[edge_features_agg].values),dtype = torch.long)

                # Define tensor of edge features
                edge_num_feach = 1
                edge_attr = torch.from_numpy(np.ones(shape = ((edge_index.size()[1]), edge_num_feach)))
                edge_attr = torch.tensor(edge_attr,dtype = torch.long)

                # Define tensor of targets
    #             y = torch.from_numpy(data_row_gran['time'].values)
                y = torch.tensor(target_val,dtype = torch.long)


                data_graph = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y=y)

                data_list.append(data_graph)

#             print("--- %s seconds ---" % (time.time() - start_time))
            print('end total')
            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))