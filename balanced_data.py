import numpy as np
import pandas as pd
from collections import Counter
from random import shuffle
import cv2

train_data = np.load('data/training_data.npy')

print(len(train_data))

df = pd.DataFrame(train_data)

print(df.head())
print(Counter(df[1].apply(str)))

lefts = []
right = []
forwards = []

# shuffle(train_data)

for index, data in enumerate(train_data):
    img = data[0]
    choice = data[1]

    # if choice == [1, 0, 0]:   training data noise
    if choice == [1, 1, 0]:
        # lefts.append([img, choice]) remove noise
        lefts.append([img, choice])
        # print("left direction!")

    if choice == [0, 1, 0]:
        forwards.append([img, choice])
        # print("forward!")

    # if choice == [0, 0, 1]:   training data noise
    if choice == [0, 1, 1]:
        right.append([img, choice])
        #print("right direction!")

    else:
        print("no match!")
'''
forwards = forwards[:len(lefts)][:len(right)]
lefts = lefts[:len(forwards)]
right = right[:len(right)]
'''
final_data = forwards + lefts + right


# shuffle(final_data)
print(len(final_data))
df = pd.DataFrame(final_data)
print(df.head())
print(Counter(df[1].apply(str)))

'''
np.save('data/training_data_v2.npy', final_data)
'''

# training_data shuffled imshow





for index, data in enumerate(lefts):
    img = data[0]
    cv2.imwrite('images/lefts/frame_{}.png'.format(index), img)

for index, data in enumerate(right):
    img = data[0]
    cv2.imwrite('images/rights/frame_{}.png'.format(index), img)

for index, data in enumerate(forwards):
    img = data[0]
    cv2.imwrite('images/forwards/frame_{}.png'.format(index), img)
