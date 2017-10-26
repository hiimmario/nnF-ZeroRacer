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

shuffle(train_data)

for data in  train_data:
    img = data[0]
    choice = data[1]

    if choice == [1, 0, 0]:
        lefts.append([img, choice])

    if choice == [0, 1, 0]:
        forwards.append([img, choice])

    if choice == [0, 0, 1]:
        right.append([img, choice])

    else:
        print("no match!")

forwards = forwards[:len(lefts)][:len(right)]
lefts = lefts[:len(forwards)]
right = right[:len(right)]

final_data = forwards + lefts + right

shuffle(final_data)
print(len(final_data))

np.save('data/training_data_v2.npy', final_data)


for data in train_data:
    img = data[0]
    choice = data[1]
    cv2.imshow('test', img)
    print(choice)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break

