from fastapi import FastAPI, WebSocket
from track_10102 import track_data, country_balls_amount
import asyncio
import glob
import numpy as np


from deepsort.tracker import DeepSortTracker
from current_detection import Detection


app = FastAPI(title='Tracker assignment')
imgs = glob.glob('imgs/*')
country_balls = [{'cb_id': x, 'img': imgs[x % len(imgs)]} for x in range(country_balls_amount)]
traker = []
new_id = 0
print('Started')


def euclid_distance(b1, b2):
    x1, y1 = (b1[0] + b1[2]) // 2, (b1[1] + b1[3]) // 2
    x2, y2 = (b2[0] + b2[2]) // 2, (b2[1] + b2[3]) // 2
    distance = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5
    return distance

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

    frame_data = el["data"]
    frame_id = el["frame_id"]

    if frame_id == 1:
        for obj in frame_data:
            global new_id
            obj["track_id"] = new_id
            new_id += 1
        traker.append(frame_data)
        return el
    prev_frame = traker[-1]
    cur_frame = frame_data

    for obj in cur_frame:
        dist_list = []
        i_dist_list = []
        if len(obj['bounding_box']) != 0:
            for obi in prev_frame:
                if len(obi['bounding_box']) != 0:
                    index = obi['track_id']
                    distance = euclid_distance(obj['bounding_box'], obi['bounding_box'])
                    dist_list.append(distance)
                    i_d = {'index': index, 'distance': distance}
                    i_dist_list.append(i_d)
            if len(dist_list) != 0:
                min_dist = min(dist_list)
                for i in i_dist_list:
                    if i['distance'] == min_dist and i['distance'] < 160:
                        a = i['index']
                        obj['track_id'] = a
                        for ob in prev_frame:
                            if ob['track_id'] == a:
                                prev_frame.remove(ob)
    for obj in cur_frame:
        if obj["track_id"] is None and len(obj["bounding_box"]) != 0:
            obj["track_id"] = new_id
            new_id += 1
    traker.append(cur_frame)
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
        el = tracker_soft(el)
        # TODO: part 2
        # el = tracker_strong(el)
        # отправка информации по фрейму
        await websocket.send_json(el)
    print('Accuracy:', calc_tracker_metrics(track_data))
    print('Bye..')
