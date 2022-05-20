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




# def create_label(data, split):
    
#     if split == 'train':
#         edge_label = torch.ones(data.edge_index.size(1))
#         data.edge_label = edge_label
#         data.edge_label_index = data.edge_index
    
#     if split == 'test' or split == 'val':
#         num_neg = data.edge_index.size(1)
#         neg_edge_index = negative_sampling(add_self_loops(data.edge_index)[0], num_nodes=data.num_nodes,num_neg_samples=num_neg, method='sparse')
        
#         edge_label = torch.ones(data.edge_index.size(1))
#         neg_edge_label = edge_label.new_zeros(neg_edge_index.size(1))
        
#         edge_label = torch.cat([edge_label, neg_edge_label], dim=0)
#         edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
            
#         data.edge_label = edge_label
#         data.edge_label_index = edge_index
    
#     return data
        
        

# class geo_StackOverData(InMemoryDataset):
    
#     url = 'https://snap.stanford.edu/data/sx-stackoverflow.txt.gz'
# #     raw_dir = '/home/jovyan/'
    
#     def __init__(self, root, transform=None, pre_transform=None, split = 'train'):
#         super(geo_StackOverData, self).__init__(root, transform, pre_transform)
#         print(self.processed_paths[0])
#         self.data, self.slices = torch.load(self.processed_dir + '/' + f'{split}.pt')
# #         self.raw_dir = '/home/jovyan/'
        
#     @property
#     def raw_dir(self) -> str:
#         return '/home/jovyan/'
    
    
#     @property
#     def raw_file_names(self):
#         return ['sx-stackoverflow.txt']

#     @property
#     def processed_file_names(self):
#         return ['train.pt', 'test.pt', 'val.pt']
    
#     @property
#     def processed_dir(self):
#         return osp.join(self.root, 'processed', 'data_stackoverflow_v1')

#     def download(self):
#         path = download_url(self.url, self.raw_dir)

#     def process(self):
        
#         # Read data
#         data = pd.read_csv(osp.join('/home/jovyan/', 'sx-stackoverflow.txt'), header=None, sep = ' ')
#         data = data[:1000000]
#         data = data.rename(columns = {0:'source', 1:'target', 2:'timestamp'})
#         data = data.sort_values(by = 'timestamp')

#         data_drop_dublicates = data.drop_duplicates(subset=['source', 'target','timestamp'])
#         data_second_drop = data_drop_dublicates.drop_duplicates(subset=['source', 'target'])
#         data = data_second_drop[data_second_drop['source'] != data_second_drop['target']]
#         data = data.drop(columns = ['timestamp'])

#         train_size = 0.8
#         test_size = 0.15
#         val_size = 0.05

#         data_split_dict = dict()
#         data_split_dict['train'] = np.arange(0,int(data.shape[0]*train_size))
#         data_split_dict['test'] = np.arange(data.shape[0]*train_size,data.shape[0]*(train_size+test_size))
#         data_split_dict['val'] = np.arange(data.shape[0]*(train_size + test_size),(data.shape[0]*(train_size+test_size + val_size)))
        
# #         transform = T.Compose([
# #             T.ToDevice(device),
# #             T.RandomLinkSplit(num_val=0.05, num_test=0.1, is_undirected=False,
# #                               add_negative_train_samples=False),
# #         ])
        
#         for split in data_split_dict.keys():
# #             print(split)
#             data_split = data.iloc[data_split_dict[split], ].reset_index(drop = True)

#             total_nodes_list = list(set(list(data_split.source.values) +list(data_split.target.values)))
#             max_node = max(total_nodes_list)
#             le = preprocessing.LabelEncoder()
#             le.fit(total_nodes_list)
#             data_split['source'] = le.transform(data_split.source.values)
#             data_split['target'] = le.transform(data_split.target.values)

#             # Define tensor of edges
#             edge_index = torch.from_numpy(data_split.values.T)
#             edge_index = torch.tensor(edge_index,dtype = torch.long)

#             # Define tensor of nodes features
#             x_num_nodes = total_nodes_list
#             x_num_feach = 2
#             x = torch.from_numpy(np.ones(shape = (len(x_num_nodes), x_num_feach)))
#             x = torch.tensor(x,dtype = torch.long)

#             # Define tensor of edge features
#             edge_num_feach = 1
#             edge_attr = torch.from_numpy(np.ones(shape = ((edge_index.size()[1]), edge_num_feach)))
#             edge_attr = torch.tensor(edge_attr,dtype = torch.long)


#             # Define tensor of targets
#             y = torch.from_numpy(np.ones(shape = len(x_num_nodes)))
#             y = torch.tensor(y,dtype = torch.long)

#             # Create dataset
#             data_graph = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y=y)

#             data_list = []
#             data_list_pre = []
#             loader = GraphSAINTRandomWalkSampler(data_graph, batch_size=100, 
#                                                  walk_length=10, num_steps=10, 
#                                                  num_workers=96)
#             for grph in loader:
#                 data_list_pre.append(grph)
#                 grph = create_label(grph, split)
#                 data_list.append(grph)

# #             data_dict_total[split] = data_list
                
#             torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))


import os.path as osp

import torch
from sklearn.metrics import roc_auc_score
import dgl

import torch_geometric.transforms as T
from torch_geometric.utils import add_self_loops, negative_sampling

import pandas as pd
from sklearn import preprocessing
from torch_geometric.data import Data
from torch_geometric.utils import degree
from torch_geometric.utils import subgraph
import numpy as np

def create_label(data, split):
    
    if split == 'train':
        edge_label = torch.ones(data.edge_index.size(1))
        data.edge_label = edge_label
        data.edge_label_index = data.edge_index
    
    if split == 'test' or split == 'val':
        num_neg = data.edge_index.size(1)
        neg_edge_index = negative_sampling(add_self_loops(data.edge_index)[0], num_nodes=data.num_nodes,num_neg_samples=num_neg, method='sparse')
        
        edge_label = torch.ones(data.edge_index.size(1))
        neg_edge_label = edge_label.new_zeros(neg_edge_index.size(1))
        
        edge_label = torch.cat([edge_label, neg_edge_label], dim=0)
        edge_index = torch.cat([data.edge_index, neg_edge_index], dim=-1)
            
        data.edge_label = edge_label
        data.edge_label_index = edge_index
    
    return data



class geo_StackOverData(InMemoryDataset):
    
    url = 'https://snap.stanford.edu/data/sx-stackoverflow.txt.gz'
#     raw_dir = '/home/jovyan/'
    
    def __init__(self, root, transform=None, pre_transform=None, split = 'train'):
        super(geo_StackOverData, self).__init__(root, transform, pre_transform)
        print(self.processed_paths[0])
        self.data, self.slices = torch.load(self.processed_dir + '/' + f'{split}.pt')
#         self.raw_dir = '/home/jovyan/'
        
    @property
    def raw_dir(self) -> str:
        return '/home/jovyan/'
    
    
    @property
    def raw_file_names(self):
        return ['sx-stackoverflow.txt']

    @property
    def processed_file_names(self):
        return ['train.pt', 'test.pt', 'val.pt']
    
    @property
    def processed_dir(self):
        return osp.join(self.root, 'processed', 'data_stackoverflow_v3')

    def download(self):
        path = download_url(self.url, self.raw_dir)

    def process(self):
        
        # Read data
        data = pd.read_csv(osp.join('/home/jovyan/', 'sx-stackoverflow.txt'), header=None, sep = ' ')
        data = data.rename(columns = {0:'source', 1:'target', 2:'timestamp'})
        data = data.sort_values(by = 'timestamp')
        data = data[:1000000]

        data_drop_dublicates = data.drop_duplicates(subset=['source', 'target','timestamp'])
        data = data_drop_dublicates[data_drop_dublicates['source'] != data_drop_dublicates['target']]

        total_nodes_list = list(set(list(data.source.values) +list(data.target.values)))
        le = preprocessing.LabelEncoder()
        le.fit(total_nodes_list)
        data['source'] = le.transform(data.source.values)
        data['target'] = le.transform(data.target.values)
        total_nodes_list = list(set(list(data.source.values) +list(data.target.values)))


        source_count = data.groupby('source').count().to_dict()['target']
        target_count = data.groupby('target').count().to_dict()['source']
        all_nodes = list(set(list(source_count.keys()) + (list(target_count.keys()))))

        nodes_count_dict = dict()

        for num in all_nodes:
            nodes_count_dict[num] = 0

        for num in all_nodes:
            if num in source_count.keys() and num not in target_count.keys():
                nodes_count_dict[num] = nodes_count_dict[num] + source_count[num]
            if num not in source_count.keys() and num in target_count.keys():
                nodes_count_dict[num] = nodes_count_dict[num] + target_count[num]
            if num in source_count.keys() and num in target_count.keys():
                nodes_count_dict[num] = nodes_count_dict[num] + source_count[num] + target_count[num]

        nodes_table = pd.DataFrame()
        nodes_table['nodes'] = list(range(0,len(total_nodes_list)))
        nodes_table['count_nodes'] = nodes_table['nodes'].map(nodes_count_dict)
        deg = degree(torch.tensor(torch.from_numpy(data[['source','target']].values.T))[0], len(total_nodes_list))
        nodes_table['count_heigb'] = deg.tolist()

        data['edge_count'] = data.groupby(['source','target'])['timestamp'].transform("count")
        data_second_drop = data.drop_duplicates(subset=['source', 'target']).reset_index(drop = True)
        # data = data_second_drop.drop(columns = ['timestamp'])

        edge_index = torch.tensor(torch.from_numpy(data[['source','target']].values.T),dtype = torch.long)
        x = torch.tensor(torch.from_numpy(nodes_table[['count_nodes','count_heigb']].values),dtype = torch.long)
        edge_attr = torch.tensor(torch.from_numpy(data['edge_count'].values),dtype = torch.long).reshape(-1,1)

         # Define tensor of targets
        y = torch.from_numpy(np.ones(shape = len(total_nodes_list)))
        y = torch.tensor(y,dtype = torch.long)

        # Create dataset
        data_graph = Data(x=x, edge_index = edge_index, edge_attr = edge_attr, y=y)

        train_size = 0.7
        test_size = 0.29
        val_size = 0.01

        data_split_dict = dict()
        data_split_dict['train'] = np.arange(0,int(data.shape[0]*train_size))
        data_split_dict['test'] = np.arange(int(data.shape[0]*train_size),int(data.shape[0]*(train_size+test_size)))
        data_split_dict['val'] = np.arange(int(data.shape[0]*(train_size + test_size)),int((data.shape[0]*(train_size+test_size + val_size))))

        data_dict_total = dict()
        for split in data_split_dict.keys():  
            data_split = data.iloc[data_split_dict[split], ].reset_index(drop = True)

            total_nodes_list = list(set(list(data_split.source.values) +list(data_split.target.values)))

            le = preprocessing.LabelEncoder()
            le.fit(total_nodes_list)
            data_split['source'] = le.transform(data_split.source.values)
            data_split['target'] = le.transform(data_split.target.values)

            source_count = data.groupby('source').count().to_dict()['target']
            target_count = data.groupby('target').count().to_dict()['source']
            all_nodes = list(set(list(source_count.keys()) + (list(target_count.keys()))))
            total_nodes_list = list(set(list(data_split.source.values) +list(data_split.target.values)))

            nodes_count_dict = dict()
            for num in all_nodes:
                nodes_count_dict[num] = 0

            for num in all_nodes:
                if num in source_count.keys() and num not in target_count.keys():
                    nodes_count_dict[num] = nodes_count_dict[num] + source_count[num]
                if num not in source_count.keys() and num in target_count.keys():
                    nodes_count_dict[num] = nodes_count_dict[num] + target_count[num]
                if num in source_count.keys() and num in target_count.keys():
                    nodes_count_dict[num] = nodes_count_dict[num] + source_count[num] + target_count[num]

            nodes_table_split = pd.DataFrame()
            nodes_table_split['nodes'] = list(range(0,len(total_nodes_list)))
            nodes_table_split['count_nodes'] = nodes_table_split['nodes'].map(nodes_count_dict)
            deg_split = degree(torch.tensor(torch.from_numpy(data_split[['source','target']].values.T))[0], len(total_nodes_list))
            nodes_table_split['count_heigb'] = deg_split.tolist()

            # Define tensor of edges
            data_split['edge_count'] = data_split.groupby(['source','target'])['timestamp'].transform("count")
            edge_index = torch.tensor(torch.from_numpy(data_split[['source','target']].values.T),dtype = torch.long)


            # Define tensor of nodes features
            x = torch.tensor(torch.from_numpy(nodes_table_split[['count_nodes','count_heigb']].values),dtype = torch.long)

            # Define tensor of edge features
            edge_attr = torch.tensor(torch.from_numpy(data_split['edge_count'].values),dtype = torch.long).reshape(-1,1)


            # Define tensor of targets
            y = torch.from_numpy(np.ones(shape = len(total_nodes_list)))
            y = torch.tensor(y,dtype = torch.long)

            # Create dataset
            g = dgl.graph((edge_index.tolist()[0], edge_index.tolist()[1]))
            g.ndata['x'] = x
            g.ndata['y'] = y
            g.edata['edge_attr'] = edge_attr.reshape(1,-1)[0]
            train_eid_dict = g.edges(form='eid')

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            print('st')
            dataloader = dgl.dataloading.EdgeDataLoader(
                g, train_eid_dict, sampler,
                negative_sampler=dgl.dataloading.negative_sampler.Uniform(1),
                batch_size=25,
                num_workers=96)
            print('end')
            data_list = []
            kk = 0
            for grph in dataloader:
                kk = kk +1 
                print(kk)
                all_graph = grph[3][0]
                pos_graph = grph[1]
                neg_graph = grph[2]

                x_geo = all_graph.ndata['x']['_N']
                y_geo = all_graph.ndata['y']['_N']
                edge_index_geo = torch.tensor((all_graph.edges()[0].tolist() , all_graph.edges()[1].tolist()))
                edge_attr_geo = all_graph.edata['edge_attr'].reshape(-1,1)

                edge_label_pos_geo = torch.ones(pos_graph.edges()[0].size()[0])
                edge_label_index_pos_geo = torch.tensor([pos_graph.edges()[0].tolist(),pos_graph.edges()[1].tolist()])

                edge_label_neg_geo = torch.zeros(neg_graph.edges()[0].size()[0])
                edge_label_index_neg_geo = torch.tensor([neg_graph.edges()[0].tolist(),neg_graph.edges()[1].tolist()])

                data_graph_new = Data(x=x_geo, edge_index = edge_index_geo, edge_attr = edge_attr_geo, y=y_geo, 
                          edge_label_pos = edge_label_pos_geo, edge_label_index_pos = edge_label_index_pos_geo,
                         edge_label_neg = edge_label_neg_geo, edge_label_index_neg = edge_label_index_neg_geo)

                data_list.append(data_graph_new)
            print('save')
            print(split)
            torch.save(self.collate(data_list), osp.join(self.processed_dir, f'{split}.pt'))

                