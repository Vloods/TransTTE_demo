import sys
import os
import bs4
from pathlib import Path
import time
from datetime import datetime
import pickle

import requests
import pandas as pd

from inference_ETA import node_data

BASE = Path(os.path.realpath(__file__)).parent
with open(BASE / '../backend/app/data/average_speed.pkl', 'rb') as f:
    average_speed = pickle.load(f)

clear_nodes = pickle.load(open(BASE / '../backend/app/data/clear_nodes.pkl', 'rb'))


id2newid={}
newid2id={}
for i in range(len(clear_nodes)):
    id2newid[int(clear_nodes[i]['id'])] = int(i)
    newid2id[int(i)] = int(clear_nodes[i]['id'])


def get_weather():
    df = pd.read_csv(BASE / "../backend/app/data/meteoData.csv", sep = ";")
    weather = df.iloc[-1, :]

    clouds_moscow_day = weather["cloud"]
    velocity_moscow_day = weather["windSpeed"]
    snow_moscow_day = weather["weather"]
    temp_moscow_day = weather["temp"]
    wind_moscow_day = weather["windDir(from)"]
    press_moscow_day = weather["pressure"]

    clouds_moscow_night = weather["cloud.1"]
    velocity_moscow_night = weather["windSpeed.1"]
    snow_moscow_night = weather["weather.1"]
    temp_moscow_night = weather["temp.1"]
    wind_moscow_night = weather["windDir(from).1"]
    press_moscow_night = weather["pressure.1"]

    def weather_merger(day, night):
        result = []
        result.append(day)
        result.append(night)
        return result

    clouds_moscow = weather_merger(clouds_moscow_day, clouds_moscow_night)
    velocity_moscow = weather_merger(velocity_moscow_day, velocity_moscow_night) 
    snow_moscow = weather_merger(snow_moscow_day, snow_moscow_night)
    temp_moscow = weather_merger(temp_moscow_day, temp_moscow_night)
    wind_moscow = weather_merger(wind_moscow_day, wind_moscow_night)
    press_moscow = weather_merger(press_moscow_day, press_moscow_night)

    props = ['clouds', 'snow', 'temperature', 'wind', 'pressure', 'velocity']
    vals_moscow = [clouds_moscow, snow_moscow, temp_moscow, wind_moscow, press_moscow, velocity_moscow]

    def weather_period(clear, prop, values):
        clear[prop] = -1    

        night = pd.DatetimeIndex(clear['start_timestamp']).indexer_between_time('00:00', '11:00')
        day = pd.DatetimeIndex(clear['start_timestamp']).indexer_between_time('11:00', '19:00')
        evening = pd.DatetimeIndex(clear['start_timestamp']).indexer_between_time('19:00', '06:00')
        parts = [night, day, evening]
    #    print(parts)
    #    print(values)
    #    print(prop)
        v_index = 0
        for i in range(len(parts)):
            tmp = -1
            if (i % 2 == 0):
                tmp = 1
            else:
                tmp = 0
            clear.loc[parts[0], prop] = values[tmp]
        return clear

    now = datetime.now()
    routes = pd.DataFrame([str(now)], columns = ['start_timestamp'])

    for i in range(len(props)):
        routes = weather_period(routes, props[i], vals_moscow[i])

    return routes

def get_time_features():
    def day_period(routes):
        index = pd.DatetimeIndex(routes['start_timestamp'])
        routes['day_period'] = -1
        routes.loc[index.indexer_between_time('00:00', '06:00'), 'day_period'] = 0
        routes.loc[index.indexer_between_time('06:00', '11:00'), 'day_period'] = 1
        routes.loc[index.indexer_between_time('11:00', '19:00'), 'day_period'] = 2
        routes.loc[index.indexer_between_time('19:00', '00:00'), 'day_period'] = 3
        return routes

    now = datetime.now()
    routes = pd.DataFrame([str(now)], columns = ['start_timestamp'])
    day_period_value = day_period(routes)["day_period"][0]
    week_period = -1

    if (0 <= datetime.today().weekday() <= 4):
        week_period = 0
    else:
        week_period = 1
    return [day_period_value, week_period]


def get_average_speed(edges):
    sum_ = 0
    count_ = 0
    for edge_id in edges:
        if edge_id in average_speed:
            count_ += 1
            sum_ += average_speed[edge_id]
    if count_ == 0:
        return 15
    return sum_ / count_

def get_eta_determin(edges):
    time_ = 0
    for edge_id in edges:
        dist_ = node_data.iloc[edge_id].dist
        speed = average_speed.get(edge_id, 15)
        if speed <= 0:
            speed = 1
        time_ += dist_ / speed
    return time_


def get_sample(edges):
    names = ['mean_wheel', 'day_period', 'week_period', 'clouds', 'snow', 'temperature', 'wind', 'pressure',
             'wind_speed', 'edges']
    velocity_feature = get_average_speed(edges)
    weather_features = get_weather().values.tolist()[0][1:]
    time_features = get_time_features()
    sample = [velocity_feature] + time_features + weather_features + [edges]
    sample_dict = {names[i]: sample[i] for i in range(len(sample))}
    return [sample_dict]


if __name__ == "__main__":
    local_edges = [226910, 226911, 474256, 474255, 474254, 474253, 474252, 474251, 608346, 608345, 608344, 474244, 254916, 474243, 845350, 845361, 845362, 845363, 254798, 144800, 144801, 114363, 144799, 144798, 144797, 121286, 607850, 607849, 144805, 144806, 144807, 144808, 144809, 144810, 329343, 329344, 431796, 431795, 431794, 431793, 2167, 2166, 2165, 2164, 2163, 2162, 2161, 2160, 2159, 2158, 2157, 885, 945363, 945364, 329234, 329235, 329236, 329370, 329351, 329352, 329259, 329371, 329372, 329373, 329374, 329375, 329376, 338198, 329257, 255446, 255447, 255424, 255508, 255398, 255505, 255429, 255439, 192632, 192633, 192634, 192635, 192636, 192637, 338248, 338249, 338250, 338272, 338273, 338274, 338275, 397763, 338227, 338226, 144794, 144795, 176303, 176304, 176305, 176306, 176307, 176308, 107383, 107478, 255110, 255109, 672939, 672938, 672937, 672936, 255156, 255155, 255154, 255153, 677345, 677344, 672947, 672948, 672949, 672950, 672951, 672952, 677727, 677728, 159675, 159676, 159677, 358102, 358101, 358100, 358099, 358098, 358097, 358096, 358095, 358094, 358093, 556012, 556013, 556014, 556015, 556016, 161336, 357920, 357921, 357749, 159595, 159596, 159597, 159598, 159241, 357864, 357865, 357724, 157061, 157062, 157063, 161322, 358055, 358056, 357824, 673066, 673065, 673064, 159093, 358047, 358048, 357820, 159242, 159243, 159244, 159245, 159246, 159247, 159248, 159249, 159250, 673090, 673091, 355372, 719638, 719639, 719640, 757497, 757442, 757441, 757440, 757435, 757434, 757433, 757432, 757431, 757495, 716009, 716008, 716007, 357781, 131083, 131082, 131081, 131080, 131079, 131063, 131062, 131061, 131090, 741071, 197790, 197791, 197792, 197793, 357729, 274431, 197942, 357926, 357927, 357763, 197743, 160394, 160393, 160392, 160391, 160390, 160389, 160388, 160387, 134012, 357914, 357915, 357985, 286622, 286621, 286620, 286619, 125135, 125108, 124273, 125143, 125062, 160116, 160396, 160395, 160152, 160153, 160154, 160415, 160416, 160417, 160418, 160419, 160420, 160421, 160422, 160423, 160424, 160425, 228163, 228162, 228161, 228160, 228159, 228158, 228157, 228156, 453111, 453112, 453127, 453128, 695887, 453130, 453131, 453132, 473728, 453140, 473724, 223828, 223827, 223826, 225609, 135397, 135403, 135404, 269626, 269625, 136815, 225904, 225905, 225906, 225907, 370302, 370297, 370456, 370457, 979189, 213513, 213512, 213511, 213510, 213509, 213508, 979188, 979187, 979186, 979180, 979179, 225861, 238695, 238698, 238678, 238732, 238685, 238694, 238726, 238676, 238688, 238748, 238747, 238746]
    print(get_eta_determin(local_edges))

    # print(get_sample(local_edges))

