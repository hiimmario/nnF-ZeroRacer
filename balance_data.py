import numpy as np
import pandas as pd
from collections import Counter
import cv2
from process_image import process_image
from random import randint

train_data = np.load('data/training_data.npy')

print(len(train_data))

df = pd.DataFrame(train_data)
print(Counter(df[1].apply(str)))

lefts = []
right = []
forwards = []

for index, data in enumerate(train_data):

    # process the original picture, which is only resized to whatever you want to train and test
    # only adjust process image and call it within test_model
    processed_img = process_image(data[0])
    choice = data[1]

    # if choice == [1, 0, 0]:   training data noise
    if choice == [1, 1, 0] or choice == [1, 0, 0]:
        lefts.append([processed_img, [1, 0, 0]])
        cv2.imwrite('images_smaller/lefts/frame_{}.png'.format(index), processed_img)


    elif choice == [0, 1, 0]:
        forwards.append([processed_img, choice])
        cv2.imwrite('images_smaller/forwards/frame_{}.png'.format(index),
                        processed_img)


    elif choice == [0, 1, 1] or choice == [0, 0, 1]:
        right.append([processed_img, [0, 0, 1]])
        cv2.imwrite('images_smaller/rights/frame_{}.png'.format(index),
                        processed_img)
    else:
        pass

# for index, data in enumerate(train_data):
#
#     # process the original picture, which is only resized to whatever you want to train and test
#     # only adjust process image and call it within test_model
#     processed_img = process_image(data[0])
#     choice = data[1]
#
#     # if choice == [1, 0, 0]:   training data noise
#     if choice == [1, 1, 0] or choice == [1, 0, 0]:
#         lefts.append([processed_img, [1, 0, 0]])
#         if randint(0,9) < 8:
#             cv2.imwrite('images_smaller/lefts/frame_{}.png'.format(index), processed_img)
#         else:
#             cv2.imwrite('images_smaller_test/lefts/frame_{}.png'.format(index),
#                         processed_img)
#
#     elif choice == [0, 1, 0]:
#         forwards.append([processed_img, choice])
#
#         if randint(0,9) < 8:
#             cv2.imwrite('images_smaller/forwards/frame_{}.png'.format(index),
#                         processed_img)
#         else:
#             cv2.imwrite('images_smaller_test/forwards/frame_{}.png'.format(index),
#                         processed_img)
#
#     elif choice == [0, 1, 1] or choice == [0, 0, 1]:
#         right.append([processed_img, [0, 0, 1]])
#
#         if randint(0,9) < 8:
#             cv2.imwrite('images_smaller/rights/frame_{}.png'.format(index),
#                         processed_img)
#         else:
#             cv2.imwrite('images_smaller_test/rights/frame_{}.png'.format(index),
#                         processed_img)
#
#     else:
#         pass

final_data = forwards + lefts + right

print(len(final_data))

df = pd.DataFrame(final_data)
print(Counter(df[1].apply(str)))

# # visiualize training data
# for img, choice in final_data:
#
#     cv2.imshow("win", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#     print(choice)
#
#     if cv2.waitKey(25) & 0xFF == ord('q'):
#         cv2.destroyAllWindows()
#         break
#
# cv2.destroyAllWindows()
