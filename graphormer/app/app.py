import random
from pathlib import Path
import os
import pickle
import io
import json
import pathlib
import sys
import uvicorn
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from loguru import logger
import torch
from evaluate_points import prepare_dataset, prepare_args, prepare_points, prepare_eval_model, prepare_eval_iterator, predict_time, prepare_task, convert_to_torch, check_town
from utils import *



BASE = str(Path(os.path.realpath(__file__)).parent)
origins = ["*"]

app = FastAPI()

preloaded_weights = True

if preloaded_weights:
    weights_abakan_link = BASE + '/weights/' + 'weights_abakan.pickle'
    weights_omsk_link = BASE + '/weights/' + 'weights_omsk.pickle'
    weights_abakan = pd.read_pickle(weights_abakan_link)
    weights_abakan = [float(x) for x in weights_abakan]
    weights_omsk = pd.read_pickle(weights_omsk_link)
    weights_omsk = [float(x) for x in weights_omsk]
    weights_dict = dict()
    weights_dict['abakan'] = weights_abakan
    weights_dict['omsk'] = weights_omsk
else:
    ##### PREPARE ABAKAN ######
    dataset_name = 'abakan'
    dataset_abakan_link = BASE + '/datasets/' + dataset_name + '/raw/final.pickle'
    model_abakan_link = BASE + '/models/'+ dataset_name + '/checkpoint_last.pt'
    prepared_dataset_edge_abakan = prepare_raw_dataset_edge(dataset_name)
    prepared_dataset_edge_abakan = prepared_dataset_edge_abakan.drop_duplicates(subset = ['source', 'target']).reset_index(drop = True)
    prepared_dataset_node_abakan = prepare_raw_dataset_node(dataset_name)
    all_graph_abakan = full_geo_Abakan(prepared_dataset_edge_abakan, prepared_dataset_node_abakan)
    all_graph_abakan = all_graph_abakan.process()
    raw_edges_abakan = list(pd.read_pickle('datasets/abakan/raw/all_roads_graph.pickle').to_networkx().edges())
    args_abakan = prepare_args('abakan')
    cfg_abakan = convert_namespace_to_omegaconf(args_abakan)
    model_abakan = prepare_eval_model(args_abakan, model_abakan_link)
    task_abakan = prepare_task(args_abakan)
    #########################


    ##### PREPARE OMSK ######
    dataset_name = 'omsk'
    dataset_omsk_link = BASE + '/datasets/' + dataset_name + '/raw/final.pickle'
    model_omsk_link = BASE + '/models/'+ dataset_name + '/checkpoint_last.pt'
    prepared_dataset_edge_omsk = prepare_raw_dataset_edge(dataset_name)
    prepared_dataset_edge_omsk = prepared_dataset_edge_omsk.drop_duplicates(subset = ['source', 'target']).reset_index(drop = True)
    prepared_dataset_node_omsk = prepare_raw_dataset_node(dataset_name)
    all_graph_omsk = full_geo_Abakan(prepared_dataset_edge_omsk, prepared_dataset_node_omsk)
    all_graph_omsk = all_graph_omsk.process()
    raw_edges_omsk = list(pd.read_pickle('datasets/omsk/raw/all_roads_graph.pickle').to_networkx().edges())
    args_omsk = prepare_args('omsk')
    cfg_omsk = convert_namespace_to_omegaconf(args_omsk)
    model_omsk = prepare_eval_model(args_omsk, model_omsk_link)
    task_omsk = prepare_task(args_omsk)
    ##########################

    ##### load weights #######
    weights_abakan = get_weights(raw_edges_abakan, all_graph_abakan, cfg_abakan, model_abakan, task_abakan)
    weights_omsk = get_weights(raw_edges_omsk, all_graph_omsk, cfg_omsk, model_omsk, task_omsk)
    ##########################
    weights_dict = dict()
    weights_dict['abakan'] = weights_abakan
    weights_dict['omsk'] = weights_omsk
    
    
class Points(BaseModel):
    start_lat: float = 55.7809453
    start_lon: float = 37.6373427
    end_lat: float = 55.6217188
    end_lon: float = 37.49859


class PointsTyped(BaseModel):
    start_lat: float = 55.7809453
    start_lon: float = 37.6373427
    end_lat: float = 55.6217188
    end_lon: float = 37.49859
    type_: str = "beauty_dist_weights"


@app.get('/')
def ping():
    return {'ping': 'pong'}


@app.post('/get_path')
def return_weights(points: Points):
    # print('staaart')
    # if check_town(points) == 'abakan':
    #     print('abakan')
    #     weights = weights_abakan
    #     # print(weights)
    # elif check_town(points) == 'omsk':
    #     print('omsk')
    #     weights = weights_omsk
    #     # print(weights)
    # else:
    #     print('ERROR!!!')
    #     raise HTTPException(status_code=400, detail="Wrong coordinates (should be Omsk or Abakan)")
    return weights_dict


if __name__ == "__main__":
    uvicorn.run(app, host='0.0.0.0', port=3006)
