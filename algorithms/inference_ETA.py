from pathlib import Path
import os
import pickle

import pandas as pd
import numpy as np
from tensorflow.keras.utils import normalize, to_categorical
from sklearn.neighbors import BallTree, DistanceMetric
from tensorflow import keras

BASE = Path(os.path.realpath(__file__)).parent
DATA = BASE / '../backend/app/data'
node_data = pd.read_csv(DATA / 'edge_with_location.csv', index_col=0)
node_embeddings = pd.read_csv(DATA / "dgi_gs_final.csv", index_col=0)

# cors = []
# for i in tqdm(range(len(node_data))):
#     cors.append([node_data.lat.iloc[i], node_data.lon.iloc[i]])

# X = np.array([[np.radians(x) for x in list_] for list_ in cors])
# tree = BallTree(X, leaf_size=3, metric = DistanceMetric.get_metric('haversine'))

with open(DATA / 'balltree.pkl', 'rb') as f:
    tree = pickle.load(f)


# dist, idx = tree.query([cors[0]], k = 1)


def get_eta(samples, need_to_project_flag=False, model_name=DATA/"regression_prod.pt") -> float:
    def summ(first, second):
        return [x + y for x, y in zip(first, second)]
    
    def zerolistmaker(n):
        return list(np.array([0]*128))
    
    ids_to_remove = []
    
    if (need_to_project_flag == True):
        for i in tqdm(range(len(samples))):
            edges_id=[]
            ok = True
            for cor in samples[i]['cors']:
                dist, idx = tree.query([[cor[0]* np.pi / 180, cor[1]* np.pi / 180]], k = 1)
                if dist*6371000 > 100:
                    ids_to_remove.append(i)
                    ok = False
                    break
                edges_id.append(idx[0][0])
            if ok:
                samples[i]['edges'] = edges_id
        ids_to_remove = list(set(ids_to_remove))
    samples_copy = samples.copy()
    
    for i in ids_to_remove[::-1]:
        samples_copy.pop(i)
    
    if (need_to_project_flag == True):
        for samp in samples_copy:
            del samp['cors']
    
    emb_len = 128
    X_pre = samples_copy
    graph = node_data
    emb = []
    for i in tqdm(range(len(X_pre))):
        path_emb = zerolistmaker(emb_len)
        arr = X_pre[i]["edges"]
        for j in range(len(arr)):
            ind = graph[graph["edge_id"] == int(arr[j])].index
            if (len(ind) != 0):
                ind = graph[graph["edge_id"] == int(arr[j])].index[0]
            else:
#                print(str(i) + " " + str(j) + " " + arr[j] + " "+ str(graph[graph["edge_id"] == int(arr[j])].index[0]))
            path_emb = summ(node_embeddings.iloc[ind], path_emb)
        emb.append(path_emb)
        
    X_pre_c = pd.DataFrame(samples_copy)
    
    X = X_pre_c.join(pd.DataFrame(emb)).dropna().drop(["edges"], axis = 1).reset_index()
    wind_dir_classes = pd.get_dummies(X.wind, prefix='wind_dir_class')
    day_classes = pd.get_dummies(X.day_period, prefix='day_class')
    X[['day_class_0','day_class_1','day_class_2','day_class_3','wind_dir_class_0','wind_dir_class_45','wind_dir_class_135','wind_dir_class_180','wind_dir_class_270','wind_dir_class_315']] = 0
    X[day_classes.columns[0]] = day_classes[day_classes.columns[0]].values[0]
#    print(day_classes[day_classes.columns[0]].values[0])
    X[wind_dir_classes.columns[0]] = wind_dir_classes[wind_dir_classes.columns[0]]
#    print(X)    
    Y = X["time"]
    X = X.drop(["time", "index", "id", "start_timestamp", "day_period", "wind"], axis = 1)
#    print(X)
    
    model = keras.models.load_model(model_name)
    from tensorflow.keras.utils import normalize, to_categorical
    test_x = normalize(np.asarray(X), axis=1)
    print(model(test_x))
    return model(test_x)

if __name__ == '__main__':
    local_edges = [572896, 838883, 838884, 838885, 838886, 838887, 838879, 838880, 484740, 484741, 484742, 484743, 484781, 484720, 484721, 990772, 572909, 572908, 990773, 226536, 226537, 226538, 226539, 227505, 227506, 227810, 227811, 227812, 227813, 573338, 573342, 573341, 573340, 226706, 226689, 226690, 226691, 226692, 226693, 114049, 114048, 114047, 114046, 227507, 959442, 959443, 227332, 227331, 227458, 227459, 227460, 227461, 227462, 227463, 227464, 227465, 227466, 227467, 227468, 227469, 227470, 227471, 895542, 895541, 225474, 225475, 225432, 225433]
    # print(node_embeddings.iloc[572896])
    from get_samples import get_sample
    samples = get_sample(local_edges)
    # print(samples)
    print(get_eta(samples))
    # print(max(node_data.edge_id))
    # print(min(node_data.edge_id))
