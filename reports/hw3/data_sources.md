# Источники данных ДЗ №3

Проверено 2026-11-25 по официальным страницам наборов.

| Набор | Роль | Лицензия | Решение |
|---|---|---|---|
| COCO 2017 train | Дообучение YOLO11n | Аннотации CC BY 4.0, изображения — лицензии Flickr по каждому файлу | Используется |
| COCO 2017 val | Калибровка порогов и тест в исходном домене (протокол ДЗ №2) | То же | Используется |
| Open Images V7 test | Тест переноса | Аннотации CC BY 4.0 (Google LLC), изображения указаны как CC BY 2.0, статус каждого файла не гарантирован | Используется |
| ObjectNet | Тест переноса (домашние ракурсы) | CC BY 4.0 + запрет настройки моделей и обязательная красная рамка при публикации | Не используется |

## Open Images V7

- В test 125 436 изображений и 937 327 рамок по 600 классам.
- У каждого изображения есть проверенные человеком положительные и отрицательные метки уровня изображения. Отрицательная метка (`Confidence = 0`) означает, что класса на фото точно нет. В test таких меток 683 997.
- Отрицательный пример для задания — только фото с проверенной отрицательной меткой. Фото без проверенной метки класса в его оценке не участвуют. Это снимает проблему ДЗ №2, где неразмеченный предмет считался отсутствующим.
- Классы (oidv7-class-descriptions-boxable.csv):

| COCO | Open Images | MID |
|---|---|---|
| bottle | Bottle | /m/04dr76w |
| cell phone | Mobile phone | /m/050k8 |
| backpack | Backpack | /m/01940j |
| clock | Clock (+ Alarm clock, Digital clock, Wall clock как положительные) | /m/01x3z |
| vase | Vase | /m/02s195 |
| potted plant | Houseplant | /m/03fp41 |
| laptop | Laptop | /m/01c648 |
| keyboard | Computer keyboard | /m/01m2v |
| tv | Television | /m/07c52 |
| banana | Banana | /m/09qck |

Источники: [факты и лицензии](https://storage.googleapis.com/openimages/web/factsfigures_v7.html), [загрузка](https://storage.googleapis.com/openimages/web/download_v7.html), [классы](https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv).

## Почему не ObjectNet

- Один архив на 184 ГБ, частичной загрузки нет.
- Только метки классов, рамок нет, поэтому срез «основная находка / фон» по рамкам не построить.
- Нет класса potted plant, laptop представлен только раскрытым, бутылки разбиты на подтипы.

Источник: [objectnet.dev/download](https://objectnet.dev/download.html).
