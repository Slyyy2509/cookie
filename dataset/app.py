import sys
import threading
import time

import cv2
import numpy as np
import pyautogui
from pynput import keyboard
from ultralytics import YOLO

model = YOLO('best.pt')

COOKIE = 0
CLOSE_AD = 1
UP_CURSOR = 2

CONF = 0.5

COOKIE_CONF = 0.3
AD_CONF = 0.12
UP_CURSOR_CONF = 0.25

COOKIE_MIN_AREA = 0.015

COOKIE_CLICKS = 10
CLICK_INTERVAL = 0.05

IMG_SIZE = 640

pyautogui.FAILSAFE = True

STOP_KEYS = {'<ctrl>': 'Ctrl', '<alt>': 'Alt', 'q': 'Q'}
HOTKEY = '+'.join(STOP_KEYS)
HOTKEY_TEXT = '+'.join(STOP_KEYS.values())

stop_requested = threading.Event()


def _on_hotkey() -> None:
    if not stop_requested.is_set():
        print(f'\nОстановка по {HOTKEY_TEXT}')
    stop_requested.set()


def start_hotkey_listener() -> keyboard.Listener | None:
    try:
        listener = keyboard.GlobalHotKeys({HOTKEY: _on_hotkey})
    except Exception as error:
        print(f'Горячая клавиша недоступна ({error}). Остановка мышью в угол экрана.')
        return None
    listener.start()
    return listener


def get_screen():
    screenshot = np.array(pyautogui.screenshot())
    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_RGB2BGR)
    return screenshot


def find_objects(image):
    result = model(image, conf=0.01, imgsz=IMG_SIZE, verbose=False)[0]

    height, width = image.shape[:2]
    frame = height * width

    limits = {COOKIE: COOKIE_CONF, CLOSE_AD: AD_CONF, UP_CURSOR: UP_CURSOR_CONF}

    objects = {}
    scores = {}

    for box in result.boxes:
        class_id = int(box.cls[0])
        conf = float(box.conf[0])

        if conf < limits[class_id]:
            continue

        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
        area = (x2 - x1) * (y2 - y1)

        if class_id == COOKIE and area < COOKIE_MIN_AREA * frame:
            continue

        quality = area if class_id == COOKIE else conf

        if class_id in scores and scores[class_id] >= quality:
            continue

        scores[class_id] = quality
        objects[class_id] = (int((x1 + x2) / 2), int((y1 + y2) / 2))

    return objects


def click(x, y, times=1):
    pyautogui.click(x, y, clicks=times, interval=CLICK_INTERVAL)


def run_loop() -> None:
    while not stop_requested.is_set():
        objects = find_objects(get_screen())

        if CLOSE_AD in objects:
            click(*objects[CLOSE_AD])
            time.sleep(0.2)
            continue

        if UP_CURSOR in objects:
            click(*objects[UP_CURSOR])
            time.sleep(0.2)
            continue

        if COOKIE in objects:
            click(*objects[COOKIE], times=COOKIE_CLICKS)

        time.sleep(0.1)


def main() -> None:
    sys.stdout.reconfigure(errors='replace')

    listener = start_hotkey_listener()
    print(f'AI работает. Остановка: {HOTKEY_TEXT} или мышь в угол экрана')

    try:
        run_loop()
    except pyautogui.FailSafeException:
        print('\nОстановка: мышь в углу экрана')
    finally:
        if listener is not None:
            listener.stop()
        print('Остановлено.')


if __name__ == "__main__":
    main()