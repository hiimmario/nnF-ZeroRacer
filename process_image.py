import cv2
import numpy as np


def process_image(img):

    processed_img = cv2.resize(img, (192, 108))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2GRAY)
    processed_img = cv2.Canny(processed_img, threshold1=200, threshold2=300)

    # kernel = np.ones((1,1), np.uint8)
    # processed_img = cv2.dilate(processed_img, kernel, iterations=1)

    return processed_img
