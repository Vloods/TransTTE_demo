
var inputs_value = [0,0,0,0];
ymaps.ready(['AnimatedLine']).then(init);
function init(ymaps) {
    // Создаем карту.
    var myPlacemark, myMap = new ymaps.Map("map", {
        center: [53.71556, 91.42917],
        zoom: 15, color: "#f652a0"
    }, {
        searchControlProvider: 'yandex#search'
    });



    let counter = 0;
    var coords = [[0,0],[0,0]];



//var input2 = document.getElementById("1");
//alert(input2);

// Слушаем клик на карте.
    myMap.events.add('click', function (e) {
        coords[counter] = e.get('coords');
        counter+=1;



// alert(unputs_value[0])
        if (counter==1){
            var firstPoint = new ymaps.Placemark(coords[0], {iconContent: 'start'}, {
                preset: 'islands#blueStretchyIcon'
            });
            myMap.geoObjects.add(firstPoint);
        }


        // Если метка уже создана – просто передвигаем ее.
        if (myPlacemark) {
            myPlacemark.geometry.setCoordinates(coords[counter]);
        }
        // Если нет – создаем.
        else {
            myPlacemark = createPlacemark(coords[counter]);
            myMap.geoObjects.add(myPlacemark);
            // Слушаем событие окончания перетаскивания на метке.
            myPlacemark.events.add('dragend', function () {
                getAddress(myPlacemark.geometry.getCoordinates());
            });
        }
        getAddress(coords[counter]);


        if (counter==2){

            var secPoint = new ymaps.Placemark(coords[1], {iconContent: 'finish'}, {
                preset: 'islands#blueStretchyIcon'
            });
            myMap.geoObjects.add(secPoint);

//            fetch("https://itam.misis.ru:9996/get_path", {
            fetch("http://localhost:9999/get_path", {
                method: 'POST', headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    "start_lat": coords[0][0],
                    "start_lon": coords[0][1],
                    "end_lat": coords[1][0],
                    "end_lon": coords[1][1]
                })
            }).then((response) => {
                return response.json();
            }).then((data) => {
                global_data = data;
                //console.log(data[0]["path"]);
                generate_lines(data);

                if(inputs_value[1]==1 && inputs_value[3]==1){ //[0,1,0,1]
                    map_print(data, 0);}

                if(inputs_value[1]==1 && inputs_value[2]==1){ //[0,1,1,0]
                    map_print(data, 1);}

                if(inputs_value[1]==0 && inputs_value[3]==1){ //[0,0,0,1]
                    map_print(data, 2);}

                if(inputs_value[1]==1 && inputs_value[2]==0){ //[0,1,0,0]
                    map_print(data, 3);}

                if(inputs_value[0]==1 && inputs_value[1]==0){ //[1,0,0,0]
                    map_print(data, 4);}

                if(inputs_value[1]==0 && inputs_value[2]==1){ //[0,0,1,0]
                    map_print(data, 5);}

                if(inputs_value[0]==1 && inputs_value[1]==1){ //[1,1,0,0]
                    map_print(data, 6);}

                inputs_value = [0,0,0,0];

            });


            counter = 0;
            coords = [[0,0],[0,0]];

        }    });


//alert(coords);

    function generate_lines(data){


        var colors = ["#3EEAD6","#88E067","#EDD35F","#D99571","#F881DA","#8189E3","#4C4047"];

        for(let i=0; i<data.length; i++){

            animatedLine = new ymaps.AnimatedLine(data[i]["path"], {}, {
                // Задаем цвет.
                strokeColor: colors[i],
                // Задаем ширину линии.
                strokeWidth: 5,
                // Задаем длительность анимации.
                animationTime: 5
            });

            myMap.geoObjects.add(animatedLine);
            animatedLine.animate()
        }
    }



    function createPlacemark(coords) {
        return new ymaps.Placemark(coords, {
            iconCaption: 'поиск...'
        }, {
            preset: 'islands#blueMoneyCircleIcon',

        });
    }

    function getAddress(coords) {
        myPlacemark.properties.set('iconCaption', 'поиск...');
        ymaps.geocode(coords).then(function (res) {
            var firstGeoObject = res.geoObjects.get(0);

            myPlacemark.properties
                .set({
                    // Формируем строку с данными об объекте.
                    iconCaption: [
                        // Название населенного пункта или вышестоящее административно-территориальное образование.
                        firstGeoObject.getLocalities().length ? firstGeoObject.getLocalities() : firstGeoObject.getAdministrativeAreas(),
                        // Получаем путь до топонима, если метод вернул null, запрашиваем наименование здания.
                        firstGeoObject.getThoroughfare() || firstGeoObject.getPremise()
                    ].filter(Boolean).join(', '),
                    // В качестве контента балуна задаем строку с адресом объекта.
                    balloonContent: firstGeoObject.getAddressLine()
                });
        });
    }



    function map_print(data, map_num) {
        //alert("test button pushed");
        var colors = ["#3EEAD6","#88E067","#EDD35F","#D99571","#F881DA","#8189E3","#4C4047"];

        //for(let i=0; i<data.length; i++){

        animatedLine = new ymaps.AnimatedLine(data[map_num]["path"], {}, {
            // Задаем цвет.
            strokeColor: colors[map_num],
            // Задаем ширину линии.
            strokeWidth: 10,
            // Задаем длительность анимации.
            animationTime: 5
        });

        myMap.geoObjects.add(animatedLine);
        animatedLine.animate()

    }
}



function input1() {
    //alert(inputs_value[0]);
    inputs_value[0] =  1;
    //   alert(inputs_value);
}

function input2() {
    //alert(inputs_value[0]);
    inputs_value[1] =  1;
    //   alert(inputs_value);
}

function input3() {
    //alert(inputs_value[0]);
    inputs_value[2] =  1;
    //  alert(inputs_value);
}

function input4() {
    //alert(inputs_value[0]);
    inputs_value[3] =  1;
    //  alert(inputs_value);
}

function input5() {
    //alert(inputs_value[0]);
    inputs_value = [0,0,0,0];

}



ymaps.modules.define('AnimatedLine', [
    'util.defineClass',
    'Polyline',
    'vow'
], function(provide, defineClass, Polyline, vow) {
    /**
     * @fileOverview Анимированная линия.
     */
    /**
     * Создает экземпляр анимированной линии.
     * @class AnimatedLine. Представляет собой геообъект с геометрией geometry.LineString.
     * @param {Boolean} [options.animationTime = 4000] Длительность анимации.
     **/
    function AnimatedLine(geometry, properties, options) {
        AnimatedLine.superclass.constructor.call(this, geometry, properties, options);
        this._loopTime = 10;
        this._animationTime = this.options.get('animationTime', 4000);
        // Вычислим длину переданной линии.
        var distance = 0;
        var previousElem = geometry[0];
        this.geometry.getCoordinates().forEach(function(elem) {
            distance += getDistance(elem, previousElem);
            previousElem = elem;
        });
        // Вычислим минимальный интервал отрисовки.
        this._animationInterval = distance / this._animationTime * this._loopTime;
        // Создадим массив с более частым расположением промежуточных точек.
        this._smoothCoords = generateSmoothCoords(geometry, this._animationInterval);
    }
    defineClass(AnimatedLine, Polyline, {
        // Анимировать линию.
        start: function() {
            var value = 0;
            var coords = this._smoothCoords;
            var line = this;
            var loopTime = this._loopTime;
            // Будем добавлять по одной точке каждые 50 мс.
            function loop(value, currentTime, previousTime) {
                if (value < coords.length) {
                    if (!currentTime || (currentTime - previousTime) > loopTime) {
                        line.geometry.set(value, coords[value]);
                        value++;
                        previousTime = currentTime;
                    }
                    requestAnimationFrame(function(time) {
                        loop(value, time, previousTime || time)
                    });
                } else {
                    // Бросаем событие окончания отрисовки линии.
                    line.events.fire('animationfinished');
                }
            }

            loop(value);
        },
        // Убрать отрисованную линию.
        reset: function() {
            this.geometry.setCoordinates([]);
        },
        // Запустить полный цикл анимации.
        animate: function() {
            this.reset();
            this.start();
            var deferred = vow.defer();
            this.events.once('animationfinished', function() {
                deferred.resolve();
            });
            return deferred.promise();
        }

    });
    // Функция генерации частых координат по заданной линии.
    function generateSmoothCoords(coords, interval) {
        var smoothCoords = [];
        smoothCoords.push(coords[0]);
        for (var i = 1; i < coords.length; i++) {
            var difference = [coords[i][0] - coords[i - 1][0], coords[i][1] - coords[i - 1][1]];
            var maxAmount = Math.max(Math.abs(difference[0] / interval), Math.abs(difference[1] / interval));
            var minDifference = [difference[0] / maxAmount, difference[1] / maxAmount];
            var lastCoord = coords[i - 1];
            while (maxAmount > 1) {
                lastCoord = [lastCoord[0] + minDifference[0], lastCoord[1] + minDifference[1]];
                smoothCoords.push(lastCoord);
                maxAmount--;
            }
            smoothCoords.push(coords[i])
        }
        return smoothCoords;
    }
    // TODO переделать на гаверсинуса
    // Функция нахождения расстояния между двумя точками на плоскости.
    function getDistance(point1, point2) {
        return Math.sqrt(
            Math.pow((point2[0] - point1[0]), 2) +
            Math.pow((point2[1] - point1[1]), 2)
        );
    }
    provide(AnimatedLine);
});
