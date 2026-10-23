import time

import cv2
import numpy as np
import pyautogui
from ultralytics import YOLO

model = YOLO('best.pt')

# id классов из разметки:
# 0 - печенька, 1 - кнопка закрытия рекламы, 2 - улучшение курсора
COOKIE = 0
CLOSE_AD = 1
UP_CURSOR = 2

CONF = 0.5  # минимальная уверенность модели

# у каждого класса свой порог.
# главную печеньку модель видит на 0.38, поэтому порог ниже.
# мелкие падающие печеньки отсекаются не порогом, а размером (COOKIE_MIN_AREA)
# крестик модель видит слабо (0.13-0.2), поэтому порог низкий.
COOKIE_CONF = 0.3
AD_CONF = 0.12
# кнопку улучшения модель видит неуверенно (0.3), поэтому порог низкий
UP_CURSOR_CONF = 0.25

# минимальный размер печеньки как доля площади экрана.
# главная печенька занимает 3% экрана, а падающие мелкие меньше 0.5%.
# всё что мельче этого - не главная печенька, кликать туда не надо
COOKIE_MIN_AREA = 0.015

COOKIE_CLICKS = 10  # сколько раз подряд нажимаем печеньку
CLICK_INTERVAL = 0.05  # пауза между кликами, секунды

# размер, на котором работает модель. Должно совпадать с imgsz в train.py
IMG_SIZE = 640

# если резко дёрнуть мышь в угол экрана - программа остановится
pyautogui.FAILSAFE = True


def get_screen():
    # делаем скриншот и переводим его в тот формат, который понимает YOLO
    screenshot = np.array(pyautogui.screenshot())
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot


def find_objects(image):
    # ищем на картинке все объекты сразу, за один проход модели
    # возвращаем словарь: id класса -> точка для клика
    # verbose=False убирает лишний вывод от ultralytics в консоль
    result = model(image, conf=0.01, imgsz=IMG_SIZE, verbose=False)[0]

    # площадь всего экрана, нужна чтобы понимать какая доля кадра занята
    height, width = image.shape[:2]
    frame = height * width

    # минимальная уверенность для каждого класса
    limits = {COOKIE: COOKIE_CONF, CLOSE_AD: AD_CONF, UP_CURSOR: UP_CURSOR_CONF}

    objects = {}  # класс -> точка для клика
    scores = {}  # класс -> насколько хорош выбранный бокс

    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        # если модель не уверена - пропускаем этот бокс
        if conf < limits[class_id]:
            continue

        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        # падающие мелкие печеньки пропускаем, нам нужна только главная
        if class_id == COOKIE and area < COOKIE_MIN_AREA * frame:
            continue

        # печеньку выбираем по размеру: главная крупная.
        # остальные классы выбираем по уверенности
        quality = area if class_id == COOKIE else conf

        # оставляем только лучший бокс, иначе клик уходит не туда
        if class_id in scores and scores[class_id] >= quality:
            continue

        scores[class_id] = quality
        objects[class_id] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    return objects


def click(x, y, times=1):
    # кликаем в точку. times - сколько раз подряд
    pyautogui.click(x, y, clicks=times, interval=CLICK_INTERVAL)


def main():
    while True:
        objects = find_objects(get_screen())

        # 1. реклама - если появилась, сразу закрываем
        if CLOSE_AD in objects:
            click(*objects[CLOSE_AD])
            # print("закрыл рекламу")
            time.sleep(0.2)  # ждём, пока реклама закроется
            continue

        # 2. улучшение курсора - нажимаем один раз
        # это кнопка покупки, поэтому только один клик за цикл
        if UP_CURSOR in objects:
            click(*objects[UP_CURSOR])
            # print("улучшил курсор")
            time.sleep(0.2)  # даём кнопке исчезнуть после покупки
            continue

        # 3. печенька - нажимаем пока не мешает реклама
        if COOKIE in objects:
            click(*objects[COOKIE], times=COOKIE_CLICKS)
            # print("нажал печеньку")

        time.sleep(0.1)


if __name__ == "__main__":
    main()