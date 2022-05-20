# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from typing import Optional
from torch_geometric.datasets import *
from torch_geometric.data import Dataset
from .pyg_dataset import GraphormerPYGDataset
import torch.distributed as dist
from .mydata_abakan import geo_Abakan
from .mydata_omsk import geo_Omsk
from .mydata_stackoverflow import geo_StackOverData

class MyQM7b(QM7b):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM7b, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyQM9(QM9):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyQM9, self).process()
        if dist.is_initialized():
            dist.barrier()
            
class MyZINC(ZINC):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyZINC, self).process()
        if dist.is_initialized():
            dist.barrier()

class MyAbakanData(geo_Abakan):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyAbakanData, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyAbakanData, self).process()
        if dist.is_initialized():
            dist.barrier()
            
class MyOmskData(geo_Omsk):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyOmskData, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyOmskData, self).process()
        if dist.is_initialized():
            dist.barrier()
            
class MyStackOverData(geo_StackOverData):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyStackOverData, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyStackOverData, self).process()
        if dist.is_initialized():
            dist.barrier()


class MyMoleculeNet(MoleculeNet):
    def download(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).download()
        if dist.is_initialized():
            dist.barrier()

    def process(self):
        if not dist.is_initialized() or dist.get_rank() == 0:
            super(MyMoleculeNet, self).process()
        if dist.is_initialized():
            dist.barrier()



class PYGDatasetLookupTable:
    @staticmethod
    def GetPYGDataset(dataset_spec: str, seed: int) -> Optional[Dataset]:
        split_result = dataset_spec.split(":")
        if len(split_result) == 2:
            name, params = split_result[0], split_result[1]
            params = params.split(",")
        elif len(split_result) == 1:
            name = dataset_spec
            params = []
        inner_dataset = None
        num_class = 1

        train_set = None
        valid_set = None
        test_set = None
        
        if name == 'omsk':
            root = '/home/jovyan/graphormer_v2/examples/georides/omsk/dataset/omsk'
        if name == 'abakan':
            root = '/home/jovyan/graphormer_v2/examples/georides/dataset/abakan'
        # root = "dataset/" + name 
        if name == "qm7b":
            inner_dataset = MyQM7b(root=root)
        elif name == "qm9":
            inner_dataset = MyQM9(root=root)
        elif name == "zinc":
            inner_dataset = MyZINC(root=root)
            train_set = MyZINC(root=root, split="train")
            valid_set = MyZINC(root=root, split="val")
            test_set = MyZINC(root=root, split="test")
        elif name == "abakan":
            inner_dataset = MyAbakanData(root=root)
            train_set = MyAbakanData(root=root, split="train")
            valid_set = MyAbakanData(root=root, split="val")
            test_set = MyAbakanData(root=root, split="test")
        elif name == "omsk":
            inner_dataset = MyOmskData(root=root)
            train_set = MyOmskData(root=root, split="train")
            valid_set = MyOmskData(root=root, split="val")
            test_set = MyOmskData(root=root, split="test")
        elif name == "stackoverflow":
            inner_dataset = MyStackOverData(root=root)
            train_set = MyStackOverData(root=root, split="train")
            valid_set = MyStackOverData(root=root, split="val")
            test_set = MyStackOverData(root=root, split="test")
        elif name == "moleculenet":
            nm = None
            for param in params:
                name, value = param.split("=")
                if name == "name":
                    nm = value
            inner_dataset = MyMoleculeNet(root=root, name=nm)
        else:
            raise ValueError(f"Unknown dataset name {name} for pyg source.")
        if train_set is not None:
            return GraphormerPYGDataset(
                    None,
                    seed,
                    None,
                    None,
                    None,
                    train_set,
                    valid_set,
                    test_set,
                    name
                )
        else:
            return (
                None
                if inner_dataset is None
                else GraphormerPYGDataset(inner_dataset, seed)
            )
