import numpy as np
import cv2
import time
from direct_keys import PressKey, ReleaseKey, W, A, D
from get_keys import key_check
from grab_screen import grab_screen

from process_image import process_image

from keras.models import load_model

WIDTH = 384
HEIGHT = 216

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    PressKey(A)
    PressKey(W)
    # ReleaseKey(W)
    ReleaseKey(D)
    # ReleaseKey(A)


def right():
    PressKey(D)
    PressKey(W)
    # ReleaseKey(W)
    ReleaseKey(A)
    # ReleaseKey(D)


model = load_model("models/model13.h5")

paused = True
last_time = time.time()

for i in list(range(3))[::-1]:
    print(i + 1)
    time.sleep(1)

while True:
    if not paused:

        fps = round(1 / max((time.time() - last_time), 0.01))
        last_time = time.time()

        screen = grab_screen(region=(0, 0, 1920, 1080))

        processed_img = process_image(screen)

        model_input = processed_img.reshape(HEIGHT, WIDTH, 1)
        model_input = np.expand_dims(model_input, axis=0)

        prediction = model.predict(model_input)
        moves = np.argmax(prediction)

        print(str(moves) + " (" + str(fps) + "fps)")

        if moves == 0:
            straight()
        elif moves == 1:
            left()
        elif moves == 2:
            right()

    keys = key_check()

    # t for pause script so you can get in position again
    if 'T' in keys:
        if paused:
            paused = False
            time.sleep(1)
        else:
            paused = True
            ReleaseKey(A)
            ReleaseKey(W)
            ReleaseKey(D)
            time.sleep(1)
