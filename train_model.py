import numpy as np
import pandas as pd
import pylab
from scipy.misc import imread

from collections import Counter

from keras.models import Sequential, load_model
from keras.layers import Dense, InputLayer, Conv2D, MaxPooling2D, Flatten,\
                         Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator


# Create data generator.
batch_size = 64

train_datagen = ImageDataGenerator(
    rescale=1./255
    # rotation_range=2.,
    # width_shift_range=0.02,
    # height_shift_range=0.04
    # fill_mode='constant',
    # cval=0.,
    # channel_shift_range=80,
    # zoom_range=0.05
)

train_generator = train_datagen.flow_from_directory(
    "images",
    target_size=(150, 204),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
    # save_to_dir="temp"
)

# Modell

input_shape = (30600,)
input_reshape = (150, 204, 1)

hidden_num_units = 200
output_num_units = 3

model = Sequential([
    InputLayer(input_shape=input_reshape),
    BatchNormalization(axis=3),

    Conv2D(2 ** 4, (10, 10), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(axis=3),

    Conv2D(2 ** 5, (10, 10), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(axis=3),

    Conv2D(2 ** 6, (5, 5), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(axis=3),

    Conv2D(2 ** 7, (2, 2), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    BatchNormalization(axis=3),

    Flatten(),

    Dropout(0.2),

    Dense(units=hidden_num_units, activation='relu'),

    Dropout(0.2),

    Dense(units=output_num_units, input_dim=hidden_num_units,
          activation='softmax'),
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Callbacks
tensorboard = TensorBoard(log_dir='logs/model1', histogram_freq=0,
                          write_graph=True, write_images=False,
                          batch_size=batch_size)

# Class Weights

counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: round(max_val/num_images, 2)
                 for class_id, num_images in counter.items()}

# Train Model

nof_epochs = 7

model.fit_generator(
    train_generator,
    steps_per_epoch=10000//batch_size,
    epochs=nof_epochs,
    callbacks=[tensorboard],
    class_weight=class_weights,
    initial_epoch=0
)

model.save("models/model1.h5")

