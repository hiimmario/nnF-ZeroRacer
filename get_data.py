import cv2
import time
import numpy as np
from grab_screen import grab_screen
import os
from get_keys import key_check

def keys_to_output(keys):
    #[A, W, D]
    output = [0, 0, 0]

    if 'A'in keys:
        output[0] = 1
    if 'D'in keys:
        output[2] = 1
    if 'W' in keys:
        output[1] = 1

    return output


file_name = 'data/training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []


def main():
    paused = True
    last_time = time.time()
    while True:

        keys = key_check()

        if not paused:
            fps = round(1 / max((time.time() - last_time), 0.01))
            last_time = time.time()

            screen = grab_screen(region=(0, 0, 1920, 1080))

            # only resize the picture, processing the picture
            # while balancing the data allows to later on
            # experiment with the original (only resized) training data
            # 1920*0.2 und 1080*0.2
            # just because i can! 1060 gtx!
            processed_img = cv2.resize(screen, (384, 216))

            # visualize captured stream
            # put in comment for more frames
            # cv2.imshow("win", processed_img)
            #
            # if cv2.waitKey(25) & 0xFF == ord('q'):
            #     cv2.destroyAllWindows()
            #     break

            output = keys_to_output(keys)
            training_data.append([processed_img, output])

            print(str(fps) + "fps")

            ## if deactivated the saving process takes quiet some time
            ## but you dont miss frames while playing and saving inbetween
            # if len(training_data) % 500 == 0:
            #     print(len(training_data))
            #     np.save(file_name, training_data)

        # t for pause script so you can get in position again
        if 'T' in keys:
            if paused:
                paused = False
                time.sleep(1)
            else:
                paused = True
                time.sleep(1)
        elif 'Q' in keys:
            np.save(file_name, training_data)
            break

for i in list(range(3))[::-1]:
    print(i+1)
    time.sleep(1)

main()

# # visualize training data
#
# train_data = np.load('data/training_data.npy')
#
# for img, choice in train_data:
#
#     cv2.imshow("win", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     print(choice)
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
#
# cv2.destroyAllWindows()
