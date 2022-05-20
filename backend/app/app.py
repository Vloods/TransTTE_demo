import random
from pathlib import Path
import os
import pickle
import io

import requests
import json

import uvicorn
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
from loguru import logger

from settings import Settings
from dijkstra_inference import DijkstraPath
from eta_inference import ETAInf
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi import(
    FastAPI,
    Header,
    HTTPException,
    Depends,
    Request,
    File,
    UploadFile
    )

origins = ["*"]
BASE = Path(os.path.realpath(__file__)).parent

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount('/js', StaticFiles(directory=BASE / 'js'))
app.mount('/css', StaticFiles(directory=BASE / 'css'))

print('1')
dijkstra_abakan = DijkstraPath(BASE / 'data/dijkstra.pickle', BASE / 'data/clear_nodes.pkl')
etainf_abakan = ETAInf(BASE / 'data/SimpleTTE.pth', BASE / 'data/meteoData.csv', BASE / 'data/dgi_sage_abakan_5_5_5_relu_relu_relu_200e_mean_pool_0.0114.csv')
weights_dict_abakan = {}

print('2')
dijkstra_omsk = DijkstraPath(BASE / 'data/graph_omsk.pkl', BASE / 'data/clear_nodes_omsk.pkl')
etainf_omsk = ETAInf(BASE / 'data/SimpleTTE.pth', BASE / 'data/meteoData.csv', BASE / 'data/dgi_sage_abakan_5_5_5_relu_relu_relu_200e_mean_pool_0.0114.csv')
weights_dict_omsk = {}

print('3')

for weight in sorted((BASE / 'data/weights_abakan').iterdir()):
    if '.pkl' in weight.name:
        with open(weight, 'rb') as fd:
            weights_dict_abakan[weight.name.split('.')[0]] = pickle.load(fd)

print('4')
for weight in sorted((BASE / 'data/weights_omsk').iterdir()):
    if '.pkl' in weight.name:
        with open(weight, 'rb') as fd:
            weights_dict_omsk[weight.name.split('.')[0]] = pickle.load(fd)
    

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

def check_town(points): 
    min_N_abk = 52.84332097535
    max_N_abk = 53.9852115715
    min_E_abk = 90.91763111635001
    max_E_abk = 91.88558398090001

    min_N_omsk = 54.78700068105
    max_N_omsk = 55.39520542775
    min_E_omsk = 72.8949037781
    max_E_omsk = 73.75839039050001

    print('5')
    if min(points.start_lat, points.end_lat) >= min_N_abk and max(points.start_lat, points.end_lat) <= max_N_abk and min(points.start_lon, points.end_lon) >= min_E_abk and max(points.start_lon, points.end_lon) <= max_E_abk:
        return 'abakan'
    elif min(points.start_lat, points.end_lat) >= min_N_omsk and max(points.start_lat, points.end_lat) <= max_N_omsk and min(points.start_lon, points.end_lon) >= min_E_omsk and max(points.start_lon, points.end_lon) <= max_E_omsk:
        return 'omsk'
    else:
        return 0
    
preloaded_weights = True
print('start weight load')
if preloaded_weights:
    weights_abakan_link = str(BASE) + '/data/graphormer_weights/' + 'weights_abakan.pickle'
    weights_omsk_link = str(BASE) + '/data/graphormer_weights/' + 'weights_omsk.pickle'
    weights_abakan = pd.read_pickle(weights_abakan_link)
    weights_abakan = [float(x) for x in weights_abakan]
    weights_omsk = pd.read_pickle(weights_omsk_link)
    weights_omsk = [float(x) for x in weights_omsk]
    # weights_dict_abakan['graphormer_weights'] = weights_abakan
    # weights_dict_omsk['graphormer_weights'] = weights_omsk
else:
    a = Points()
    body = {"start_lat": a.start_lat,
                        "start_lon": a.start_lon,
                        "end_lat": a.end_lat,
                        "end_lon": a.end_lon,
                       "type": 'lolo'}

    r = requests.post('http://127.0.0.1:3006/get_path', headers = {'Content-Type': 'application/json'}, json = body)
    weights_abakan = r.json()['abakan']
    weights_omsk = r.json()['omsk']
print('end weight load')

    

@app.get('/')
def ping():
    print('1')
    html_file = BASE / 'index.html'
    return HTMLResponse(html_file.open().read())


@app.post('/get_path')
def return_path(points: Points):
    
    print('start get_path')
    logger.info(points)
    if not all(map(lambda x: 0 <= x[1] <= 180, points)):
        raise HTTPException(status_code=400, detail="Every coordinate should be in 0..180")
    cur_data_name = check_town(points)
    response = []
    paths = []
    if cur_data_name == 'abakan':
        print('abakan')
        for name, weights in weights_dict_abakan.items():
            p, path = dijkstra_abakan.get_shortest_path((points.start_lat, points.start_lon), (points.end_lat, points.end_lon), weights)
            etapred = etainf_abakan.forward(p, points, path[1][:2], path[1][-2:])
            dict_ = {'path': path[1][:-1], 'eta': etapred, 'type': name}
            paths.append(dict_['path'])
            response.append(dict_)
        name = 'graphormer_weights'
        weights = weights_abakan
        p, path, time = dijkstra_abakan.get_shortest_path_grph((points.start_lat, points.start_lon), (points.end_lat, points.end_lon), weights)
        dict_ = {'path': path[1][:-1], 'eta': int(time), 'type': name}
        response.append(dict_)
        print('len response', len(response))
    elif cur_data_name == 'omsk':
        print('omsk')
        for name, weights in weights_dict_omsk.items():
            p, path, time = dijkstra_omsk.get_shortest_path_grph((points.start_lat, points.start_lon), (points.end_lat, points.end_lon), weights)
            time = time / 10
            # time = etainf_omsk.forward(p, points, path[1][:2], path[1][-2:])
            dict_ = {'path': path[1][:-1], 'eta': int(time), 'type': name}
            # paths.append(dict_['path'])
            response.append(dict_)
        name = 'graphormer_weights'
        weights = weights_omsk
        p, path, time = dijkstra_omsk.get_shortest_path_grph((points.start_lat, points.start_lon), (points.end_lat, points.end_lon), weights)
        dict_ = {'path': path[1][:-1], 'eta': int(time), 'type': name}
        response.append(dict_)
        print('len response', len(response))
    else:
        print('error!')
        raise HTTPException(status_code=400, detail="Coordinates should be in Omsk or Abakan")
    
            
    return response


if __name__ == "__main__":
    app_name = "app:app"
    # uvicorn.run(app_name, host=Settings().hostname, port=Settings().port, log_level="info", ssl_certfile=Settings().ssl_certfile, ssl_keyfile=Settings().ssl_keyfile)
    uvicorn.run(app, host='0.0.0.0', port=80)