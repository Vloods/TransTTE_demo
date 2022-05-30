# TransTTE

![Pipeline_image](resources/transtte_pipeline_wh.png#gh-light-mode-only)
![Pipeline_image](resources/transtte_pipeline_bl.png#gh-dark-mode-only)

Welcome to the official repo of the TransTTE model -- transformer-based travel time estimation algorithm. Here we present the source code for PKDD'22 demo track paper "Logistics, Graphs, and Transformers: Towards improving Travel Time Estimation".

Natalia Semenova, Artyom Sosedka, Vladislav Tishin, Vladislav Zamkovoy, [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov)

You can access inference of our model at [transtte.online](http://transtte.online)

arXiv PDF: _to be added_.

# Prerequisites

It is possible to run Visual Tool and Graphormer locally, but we strongly recomend to use provided Dockerfiles

**Backend:**

```
fastapi==0.67.0
pydantic==1.8.2
uvicorn==0.14.0
pandas==1.3.4
sklearn==0.0
python-igraph==0.9.6
loguru==0.5.3
torch==1.9.1+cu111

```

**Model:**

```
lmdb==1.3.0
torch-scatter==2.0.9
torch-sparse==0.6.12
torch-geometric==1.7.2
tensorboardX==2.4.1
ogb==1.3.2
rdkit-pypi==2021.9.3
dgl==0.7.2
igraph==0.9.10
setuptools==0.1.96
numpy==1.20.3

```
Also you need to install [fairseq](https://github.com/facebookresearch/fairseq) to fit graphormer


# Local test

**Prepare repository, data and weights:**
- Clone repository: ```git clone https://github.com/Vloods/TransTTE_demo```
- Download [backend data](https://disk.yandex.ru/d/NHj3ukteUGn-dA) and put it in backend/app/data
- Download [graphormer models](https://disk.yandex.ru/d/rQCIJs_7Q7Li6g) and put it in graphormer/app/models
- In addition, you need to download geo-datasets to fit graphormer. However, it's a private data. Thus, if you want to fit graphormer localy, please [contact](semenova.bnl@gmail.com) us


**How to run Visual Tool:**
- Install and run Docker
- Build Docker image with backend/Dockerfile via run command ```docker build . -t visual" in terminal```
- Run Docker container via run ```docker run --rm -it -p 80:80 visual``` in terminal
- Go to http://127.0.0.1:80/ 

**How to run Graphormer:**
- Install and run Docker
- Build Docker image with graphormer/Dockerfile via run ```docker build . -t graphormer``` in terminal
- Run Docker container via run ```docker run --rm -it -p 80:80 graphormer``` in terminal
- Run python script to get times for each edge. Visual tool use this times in order to find the shortest way between two points.
     
####  Python script:
      r = requests.post('http://0.0.0.0:80/get_weights', headers = {'Content-Type': 'application/json'})
      weights_dict = r.json()


# Datasets

We provide two datasets corresponding to the cities of Abakan and Omsk. For each of these datasets, there are two types of target values -- real travel time (considered in this study) and real length of trip. 

<table>
<tr><th>Road network</th><th>Trips</th></tr>
<tr><td>

| | Abakan | Omsk |
|--|--|--|
|Nodes| 65524 | 231688 |
|Edges| 340012 |  1149492 |
|Clustering| 0.5278 | 0.53 |
|Usage median| 12 | 8 |
 
</td><td>

| | Abakan | Omsk |
|--|--|--|
|Trips number|  119986 | 120000 |
|Coverage| 0.535 |  0.392 |
|Average time| 433.61 | 622.67 |
|Average length| 3656.34 | 4268.72 |

</td></tr> </table>

Provided data could be used for research purposes only. If you want to incorporate it in your study, please send request to semenova.bnl@gmail.com.

# License

Established code released as open-source software under the MIT license.

# Contact us

If you have some questions about the code, you are welcome to open an issue, I will respond to that as soon as possible.

# Citation

```
To be added
```
