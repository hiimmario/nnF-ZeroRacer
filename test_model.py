import numpy as np
import cv2
import time
from direct_keys import PressKey, ReleaseKey, W, A, D
from get_keys import key_check
from grab_screen import grab_screen

from process_image import process_image

from keras.models import load_model

WIDTH = 192
HEIGHT = 108


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)


def left():
    PressKey(A)
    PressKey(W)
    ReleaseKey(D)


def right():
    PressKey(D)
    PressKey(W)
    ReleaseKey(A)


model = load_model("models/model-64batchsize-640hidden_units_nocanny_10k.h5")
model = load_model("checkpoints/weights.0_01-0.53.hdf5")

paused = True
last_time = time.time()

for i in list(range(3))[::-1]:
    print(i + 1)
    time.sleep(1)

loop_index = 0

while True:
    if not paused:

        fps = round(1 / max((time.time() - last_time), 0.01))
        last_time = time.time()

        screen = grab_screen(region=(375, 0, 1570, 1080))
        processed_img = cv2.resize(screen, (192, 108))
        processed_img = (process_image(processed_img)/255)

        # cv2.imwrite('images_test/frame_{}.png'.format(loop_index), processed_img)
        # cv2.imshow("Test", processed_img)

        # processed_img = cv2.imread("images_smaller_test/rights/frame_10586.png")
        # processed_img = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)

        # loop_index += 1

        model_input = processed_img.reshape(HEIGHT, WIDTH, 1)
        # cv2.imshow("Test", model_input)
        model_input = np.expand_dims(model_input, axis=0)

        prediction = model.predict(model_input)
        moves = np.argmax(prediction)

        print(str(moves) + " (" + str(fps) + "fps)")

        if moves == 0:
            # print("straight")
            straight()
        elif moves == 1:
            # print("left")
            left()
        elif moves == 2:
            # print("right")
            right()

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

    keys = key_check()

    if 'Q' in keys:
        break

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
