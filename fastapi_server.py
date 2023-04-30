from fastapi import FastAPI, WebSocket
from track_3 import track_data, country_balls_amount
import asyncio
import glob
import numpy as np
import cv2

from deepsort.tracker import DeepSortTracker
from current_detection import Detection


app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
print('Started')


def tracker_soft(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов

    Ограничения:
    - необходимо использовать как можно меньше ресурсов (представьте, что
    вы используете embedded устройство, например Raspberri Pi 2/3).
    -значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме
    """
    return el


def tracker_strong(el):
    """
    Необходимо изменить у каждого словаря в списке значение поля 'track_id' так,
    чтобы как можно более длительный период времени 'track_id' соответствовал
    одному и тому же кантри болу.

    Исходные данные: координаты рамки объектов, скриншоты прогона

    Ограничения:
    - вы можете использовать любые доступные подходы, за исключением
    откровенно читерных, как например захардкодить заранее правильные значения
    'track_id' и т.п.
    - значение по ключу 'cb_id' является служебным, служит для подсчета метрик качества
    вашего трекера, использовать его в алгоритме трекера запрещено
    - запрещается присваивать один и тот же track_id разным объектам на одном фрейме

    P.S.: если вам нужны сами фреймы, измените в index.html значение make_screenshot
    на true для первого прогона, на повторном прогоне можете читать фреймы из папки
    и по координатам вырезать необходимые регионы.
    TODO: Ужасный костыль, на следующий поток поправить
    """
    object_coord = []
    # image = cv2.imread('data/{0}.png'.format(el['frame_id']))
    # image = cv2.resize(image, (1000, 800))
    for obj in el['data']:
        if (len(obj['bounding_box']) == 0):
            continue
        tlwh = np.array([
            int(obj['bounding_box'][2] * 0.5 + 0.5 * obj['bounding_box'][0]),
            obj['bounding_box'][3],
            obj['bounding_box'][2] - obj['bounding_box'][0],
            obj['bounding_box'][3] - obj['bounding_box'][1]
        ])
        # tlwh_img = [0 if tlwh[i] < 0 else tlwh[i] for i in range(4)]
        # tlwh_img = [1000 if tlwh_img[i] > 1000 and i in (0, 2) else tlwh_img[i] for i in range(4)]
        # tlwh_img = [800 if tlwh_img[i] > 800 and i in (1, 3) else tlwh_img[i] for i in range(4)]
        # obj_image = image[tlwh_img[1]:tlwh_img[1] + tlwh_img[3], tlwh_img[0]:tlwh_img[0] + tlwh_img[2]]
        feature = None
        # if obj_image.shape[0] * obj_image.shape[1] > 0:
        #     feature = np.asarray(obj_image)
        object_coord.append(Detection(
            tlwh=tlwh,
            confidence=1,
            feature=feature,
        ))
    tracker = DeepSortTracker()
    tracker.update(object_coord)
    dont_check_count = 0
    for i, obj in enumerate(el['data']):
        if len(obj['bounding_box']) == 0:
            dont_check_count += 1
            continue
        obj['track_id'] = tracker.tracks[i - dont_check_count].track_id
    return el


def calc_tracker_metrics(track_data):
    map_id = {}
    true_track = 0
    all_track = 0
    for el in track_data:
        for obj in el['data']:
            if len(obj['bounding_box']) == 0:
                continue
            if obj['cb_id'] not in map_id.keys():
                map_id[obj['cb_id']] = obj['track_id']
            all_track += 1
            if obj['track_id'] == map_id[obj['cb_id']]:
                true_track += 1
    accuracy = true_track / all_track
    return accuracy


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    print('Accepting client connection...')
    await websocket.accept()
    # отправка служебной информации для инициализации объектов
    # класса CountryBall на фронте
    await websocket.send_text(str(country_balls))
    for el in track_data:
        await asyncio.sleep(0.5)
        # TODO: part 1
        # el = tracker_soft(el)
        # TODO: part 2
        el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
    print('Accuracy:', calc_tracker_metrics(track_data))
    print('Bye..')
