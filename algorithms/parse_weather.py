import sys
import os
import bs4
import requests
import time
from datetime import datetime
from tqdm import tqdm
import pandas as pd

citeName = 'https://www.gismeteo.ru/diary/'
headers = \
    {'User-Agent': 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) Gecko/20100101 Firefox/24.0'}



def parseData(lineData):
    return int(lineData[0].getText())


def parseTemp(lineData, shift):
    return int(lineData[1 + shift].getText())


def parsePressure(lineData, shift):
    return int(lineData[2 + shift].getText())


def parseCloud(lineData, shift):
    cloud = lineData[3 + shift].find('img').get('src').split('/'
            )[-1].split('.')[0]
    if cloud == 'sun':
        cloud = 1
    if cloud == 'sunc':
        cloud = 3
    if cloud == 'suncl':
        cloud = 5
    if cloud == 'dull':
        cloud = 10
    return cloud


def parseWeather(lineData, shift):
    weather = None
    try:
        weather = lineData[4 + shift].find('img').get('src').split('/'
                )[-1].split('.')[0]
        if weather == 'snow':
            weather = 2
        if weather == 'rain':
            weather = 1
        if weather == 'storm':
            weather = 1
    except AttributeError:
        weather = 0
    return weather


def parseWindData(lineData, shift):
    windData = lineData[5 + shift].find('span').getText()
    windSpeed = int(windData.split(' ')[-1].split(u"м")[0])
    windDir = windData.split(' ')[0]

    if windDir == u"С":
        windDir = 0
    if windDir == u"СВ":
        windDir = 45
    if windDir == u"В":
        windDir = 90
    if windDir == u"ЮВ":
        windDir = 135
    if windDir == u"Ю":
        windDir = 180
    if windDir == u"ЮЗ":
        windDir = 225
    if windDir == u"З":
        windDir = 270
    if windDir == u"СЗ":
        windDir = 315
    return (windSpeed, windDir)


def parseLine(lineData, shift):
    data = parseData(lineData)
    temp = parseTemp(lineData, shift)
    pr = parsePressure(lineData, shift)
    cloud = parseCloud(lineData, shift)
    weather = parseWeather(lineData, shift)
    (windSpeed, windDir) = parseWindData(lineData, shift)

    return {
        'data': data,
        'temp': temp,
        'pressure': pr,
        'cloud': cloud,
        'weather': weather,
        'windSpeed': windSpeed,
        'windDir': windDir,
        }


def parseTable(pageData):
    allData_tmp = []
    table = pageData.find('table')
    linesInTable = table.find_all('tr')
    linesInTable = linesInTable[2:]
    for (fullLineId, fullLine) in enumerate(linesInTable):
        try:
            day = parseLine(fullLine.find_all('td'), 0)
            night = parseLine(fullLine.find_all('td'), 5)
            allData_tmp.append([day, night])
        except ValueError:
            print ('No Data in line')
        except AttributeError:
            print ('No Data in line')
    return allData_tmp


def writeInCSV(allData):
    f = open('meteoData.csv', 'w')
    f.write('day;temp;pressure;windDir(from);windSpeed;weather;cloud;temp;pressure;windDir(from);windSpeed;weather;cloud\n'
            )
    for fullDayData in allData:
        day = fullDayData[0]
        night = fullDayData[1]
        f.write(str(fullDayData[0]['data']) + ';')
        f.write('{0};{1};{2};{3};{4};{5};'.format(
            day['temp'],
            day['pressure'],
            day['windDir'],
            day['windSpeed'],
            day['weather'],
            day['cloud'],
            ))
        f.write('{0};{1};{2};{3};{4};{5}'.format(
            night['temp'],
            night['pressure'],
            night['windDir'],
            night['windSpeed'],
            night['weather'],
            night['cloud'],
            ))
        f.write('\n')
    f.close()


def get_info():
    today = datetime.today()
    cityId = '4368/'

    allYears = [str(today.year)]
    allData = []

    month_start = today.month
    month_end = today.month

    for year in allYears:
        for mnthId in range(month_start, month_end+1):
            linkToRegion = citeName + cityId + year \
                + '/{0}/'.format(mnthId)
            print(linkToRegion)
            s = requests.get(linkToRegion, headers=headers)
            pageData = bs4.BeautifulSoup(s.text, 'html.parser')
            allData.extend(parseTable(pageData))

    writeInCSV(allData)
#    print ('Done!')


if __name__ == "__main__":
    get_info()