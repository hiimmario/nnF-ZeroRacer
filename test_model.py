import numpy as np
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, D
from getkeys import key_check
from grabscreen import grab_screen

from keras.models import load_model

WIDTH = 150
HEIGHT = 204

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    PressKey(A)
    PressKey(W)
    #ReleaseKey(W)
    ReleaseKey(D)
    #ReleaseKey(A)


def right():
    PressKey(D)
    PressKey(W)
    # ReleaseKey(W)
    ReleaseKey(A)
    # ReleaseKey(D)


model = load_model("models/model1.h5")

last_time = time.time()
for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)

paused = False

while True:
    if not paused:
        print('loop took {} seconds'.format(time.time() - last_time))
        last_time = time.time()

        screen = grab_screen(region=(100, 100, 612, 548))
        processed_img = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        # processed_img = cv2.Canny(processed_img, threshold1=200,
        #                           threshold2=300)
        #
        # kernel = np.ones((2, 2), np.uint8)
        # processed_img = cv2.dilate(processed_img, kernel, iterations=1)
        processed_img = cv2.resize(processed_img, (204, 150))

        # cv2.imshow("window1", processed_img)
        #
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

        model_input = processed_img.reshape(WIDTH, HEIGHT, 1)
        model_input = np.expand_dims(model_input, axis=0)

        prediction = model.predict(model_input)
        moves = np.argmax(prediction)

        if moves == 0:
            straight()
        elif moves == 1:
            left()
        elif moves == 2:
            right()

        # print(moves, prediction)

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
