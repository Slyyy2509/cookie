import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def _import_gui():
    """Загрузить pyautogui/pynput, объяснив причину, если X-сервер недоступен."""
    try:
        import pyautogui
        from pynput import keyboard
    except Exception as error:
        hint = ''
        if sys.platform.startswith('linux'):
            hint = (
                '\nНужен X11: переменная DISPLAY не задана или X-сервер недоступен.'
                '\nWayland не поддерживается - переключите сессию на "Ubuntu on Xorg".'
            )
        raise SystemExit(
            f'Не удалось загрузить pyautogui/pynput: '
            f'{type(error).__name__}: {error}{hint}'
        ) from error
    return pyautogui, keyboard


pyautogui, keyboard = _import_gui()

# CWD всегда = директория скрипта (как в train.py), иначе 'best.pt' не найдётся
os.chdir(Path(__file__).resolve().parent)

model = YOLO('best.pt')

COOKIE = 0
CLOSE_AD = 1
UP_CURSOR = 2

COOKIE_CONF = 0.15
AD_CONF = 0.12
UP_CURSOR_CONF = 0.25

# Печень занимает ~3% кадра, но в размеченных кадрах встречаются и 0.1%.
# Порог отсекает только совсем мелкие боксы, не мешая настоящей печеньке.
COOKIE_MIN_AREA = 0.005

COOKIE_CLICKS = 10
CLICK_INTERVAL = 0.05

# Печень на экране 1920x1080 занимает ~3% кадра. При 640 она превращается
# в ~80x85 пикселей и теряется моделью (hit-rate 64% -> 88.9% при 1152).
IMG_SIZE = 1152

pyautogui.FAILSAFE = True

# Пробел вместо Ctrl+Alt+Q: на Windows бэкенд отдаёт Ctrl как Key.ctrl_l,
# а парсер pynput ждёт Key.ctrl — такой хоткей не срабатывает никогда.
STOP_KEYS = {'<space>': 'Space'}
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

        # Реклама модальная: пока её не закрыли, клик по печенье уйдёт в рекламу.
        if CLOSE_AD in objects:
            click(*objects[CLOSE_AD])
            time.sleep(0.2)
            continue

        # Печень и апгрейд кликаются в одном кадре: апгрейд не должен
        # отнимать кадр у печенья (иначе печень кликается через раз).
        if COOKIE in objects:
            click(*objects[COOKIE], times=COOKIE_CLICKS)

        if UP_CURSOR in objects:
            click(*objects[UP_CURSOR])
            time.sleep(0.05)

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
