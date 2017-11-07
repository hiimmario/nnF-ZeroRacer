import cv2
import time
import numpy as np
from grabscreen import grab_screen
import os
from getkeys import key_check

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


# new folder!
file_name = 'data/training_data.npy'

if os.path.isfile(file_name):
    print('File exists, loading previous data!')
    training_data = list(np.load(file_name))
else:
    print('File does not exist, starting fresh!')
    training_data = []

def main():
    last_time = time.time()
    while True:
        screen = grab_screen(region=(100, 100, 612, 548))
        processed_img = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
        processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)

        kernel = np.ones((2, 2), np.uint8)
        processed_img = cv2.dilate(processed_img, kernel, iterations=1)
        processed_img = cv2.resize(processed_img, (204, 150))

        # cv2.imshow("hsdf", processed_img)
        #
        # if cv2.waitKey(25) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break

        keys = key_check()
        output = keys_to_output(keys)
        training_data.append([processed_img, output])

        print('Frame took {} seconds'.format(time.time() - last_time))
        last_time = time.time()

        if len(training_data) % 500 == 0:
            print(len(training_data))
            np.save(file_name, training_data)

for i in list(range(3))[::-1]:
    print(i+1)
    time.sleep(1)

main()
