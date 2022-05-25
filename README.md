# TransTTE

![Pipeline_image](resources/transtte_pipeline_wh.png#gh-light-mode-only)
![Pipeline_image](resources/transtte_pipeline_bl.png#gh-dark-mode-only)

Welcome to the official repo of the TransTTE model -- transformer-based travel time estimation algorithm. Here we present the source code for PKDD'22 demo track paper "Logistics, Graphs, and Transformers: Towards improving Travel Time Estimation".

Natalia Semenova, Artyom Sosedka, Vladislav Tishin, Vladislav Zamkovoy, [Vadim Porvatov](https://www.researchgate.net/profile/Vadim-Porvatov)

You can access inference of our model at [transtte.online](http://transtte.online:9103).

arXiv PDF: _to be added_.

# Prerequisites

For the backend:

```
torch==1.9.1
...
```

For the Graphormer model:
# Local test:

##    Prepare repository, data and weights:
     - Clone repository: git clone https://github.com/Vloods/TransTTE_demo
     
- Download [data](https://disk.yandex.ru/d/SrQo6jW5GS_n8A) and put it in..
- Download [weights](https://disk.yandex.ru/d/wDoXLVr0PsXg_Q) and put it in..


##    How to run Visual Tool:
     - Install and run Docker
     - Build Docker image with backend/Dockerfile using docker build . -t visual
     - Run Docker container using docker run --rm -it -p 80:80 visual
     - Go to http://127.0.0.1:80/ 

##    How to run Graphormer
     - Install and run Docker
     - Build Docker image with graphormer/Dockerfile using docker build . -t graphormer
     - Run Docker container using docker run --rm -it -p 80:80 graphormer
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

# Model running

To be defined.

# Application details

To be defined.

# License

Established code released as open-source software under the MIT license.

# Contact us

If you have some questions about the code, you are welcome to open an issue, I will respond to that as soon as possible.

# Citation

```
To be added
```

# OLD


Представленный пайплайн позволяет валидировать и уточнять алгоритмы построения кратчайшего маршрута посредством использования гибридной графовой модели машинного обучения. Основным назначением программы является оценка маршрутов для самокатов по метрикам затрачиваемого на поездку времени, живописности окружающего ландшафта и безопасности с целью улучшения пользовательского опыта.


# Структура репозитория
- `algorithms` - ноутбуки и скирпты для вызова тех или иных алгоритмов
- `preprocessing` - ноутбуки для препроцессинга данных
- `backend` - бекенд сервер на FastAPI, реализует метод генерации маршрутов и их ETA по 2м точкам
- `data` и `backend/app/data` - используемые данные: граф Москвы, матрица высот и тд. 

# Туториал
- Запустите `backend/app/dijkstra_inference.py` с координатами и требуемыми весами из `backend/app/data/weights`
- Вы получите два массива, второй - координаты путей
- Если необходимо, запустите `inference_ETA.py` по примеру для получения ETA

## Архитектура предиктивной модели ETA

Для решения задачи оценки времени прибытия используется гибридная архитектура, состоящая из графовой нейронной сети, осуществляющей построение векторных представлений путей самокатов, и регрессионной модели.
 
## Прочие метрики

Расчет метрики, оценивающей визуальное окружение во время поездки, осуществляется посредством проекции геометок смежных с улицами достопримечательностей, парковых пространств и прочих культурных площадок на транспортный граф. Впоследствии полученное таким образом признаковое описание ребер маршрута агрегируется в зависимости от персональных предпочтений пользователя в оценку потенциального пути следования.  

Метрика безопасности пути также складывается из безопасности ребер, которая получается из данных о ближайших ДТП, косвенной исторической информации о поездках самокатов на определенных участках дорог и специфики ассоциированного с элементами транспортного графа ландшафта. 

## Данные
1. Предоставленные Whoosh данные о поездках на самокатах, каршеринге и такси за Май

2. XML, GeoJSON и другие типы геоданных Москвы: [OSM](https://download.bbbike.org/osm/bbbike/Moscow/)

3. GeoJSON ДТП: [Карта ДТП](https://dtp-stat.ru/opendata/) 

Данные потяжелее лежат [ТУТ](https://drive.google.com/drive/folders/1BJzO_0bPF-TlAnkiN37OeygfrDL-NKSr?usp=sharing)

## Препроцессинг данных
1. Для получения графа Москвы, был взят [OSM XML](https://download.bbbike.org/osm/bbbike/Moscow/) и произведено:
   - удаление не нужных ноды(например, нод зданий)
   - формирование связанных с нодами рёбер путей и их фичей(например, тип поверхности, освещённость, длинна дороги и тд)
    
