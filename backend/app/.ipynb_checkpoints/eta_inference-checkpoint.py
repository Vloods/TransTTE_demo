import torch
import pandas as pd
from datetime import datetime
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import haversine_np, get_perp
from ml import FFNet

class ETAInf:
    def __init__(self, path_to_model, path_to_features, path_to_embs):
        self.model = FFNet(152, False)
        self.model.load_state_dict(torch.load(path_to_model))
        self.nodes = np.array(pd.read_csv(path_to_embs, index_col = 0))
        self.wfile = path_to_features
        self.model.eval()
        
    def preprocess_route(self, route, points, st_edge_cors, fin_edge_cors):
        start_perp = get_perp(st_edge_cors[0][0], st_edge_cors[0][1], st_edge_cors[1][0], st_edge_cors[1][1], points.start_lat, points.start_lon)
        start_perp = start_perp if start_perp is not None else [st_edge_cors[0][0], st_edge_cors[0][1]]
        dist_to_a = haversine_np(start_perp[1], start_perp[0], points.start_lon, points.start_lat)
        start_point_meters = haversine_np(start_perp[1], start_perp[0], st_edge_cors[0][1], st_edge_cors[0][0])
        start_point_part = start_point_meters / haversine_np(st_edge_cors[0][1], st_edge_cors[0][0], st_edge_cors[1][1], st_edge_cors[1][0])
        
        end_perp = get_perp(fin_edge_cors[0][0], fin_edge_cors[0][1], fin_edge_cors[1][0], fin_edge_cors[1][1], points.end_lat, points.end_lon)
        end_perp = end_perp if end_perp is not None else [fin_edge_cors[1][0], fin_edge_cors[1][1]]
        dist_to_b = haversine_np(end_perp[1], end_perp[0], points.start_lon, points.start_lat)
        finish_point_meters = haversine_np(end_perp[1], end_perp[0], fin_edge_cors[1][1], fin_edge_cors[1][0])
        finish_point_part = finish_point_meters / haversine_np(fin_edge_cors[0][1], fin_edge_cors[0][0], fin_edge_cors[1][1], fin_edge_cors[1][0])
        stat_data = pd.DataFrame({"dist_to_b": [dist_to_b], "dist_to_a": [dist_to_a], "start_point_meters": [start_point_meters], "finish_point_meters": [finish_point_meters], "start_point_part":[start_point_part], "finish_point_part": [finish_point_part]})
        
        weather = pd.read_csv(self.wfile, delimiter=";")
        today = datetime.today()
        week_period = pd.DataFrame({"week_period":[int(today.weekday() >= 5)]})
        weather_today = weather.iloc[-1,1:7][["cloud", "weather", "temp", "windSpeed", "pressure"]] if 9 < today.hour < 18 else weather.iloc[-1,7:][["cloud.1", "weather.1", "temp.1", "windSpeed.1", "pressure.1"]]
        wdt = pd.DataFrame.from_dict({'wind_dir_class_0':[0], 'wind_dir_class_45':[0], 'wind_dir_class_90':[0], 'wind_dir_class_135':[0], 'wind_dir_class_180':[0], 'wind_dir_class_225':[0], 'wind_dir_class_270':[0], 'wind_dir_class_315':[0]})
        wdt.iloc[0, int(weather.iloc[-1]["windDir(from)" if 9 < today.hour < 18 else "windDir(from).1"] / 45)] = 1
        dct = pd.DataFrame({"day_class_0":[0], "day_class_1":[0], "day_class_2":[0], "day_class_3":[0]})
        dct.iloc[0, today.hour // 6] = 1
        
        res = stat_data.join(week_period).join(weather_today.to_frame().T.reset_index().drop("index", axis=1)).join(pd.DataFrame([self.nodes[p] for p in route]).sum().to_frame().T).join(wdt).join(dct)

        return res.values
        
    def forward(self, route, points, st_edge_cors, fin_edge_cors):
        
        return int(self.model(torch.tensor(self.preprocess_route(route, points, st_edge_cors, fin_edge_cors)).float()).item())
        
    
