from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv2D, MaxPooling2D, Flatten,\
                         Dropout, BatchNormalization
from keras.callbacks import TensorBoard #, ModelCheckpoint ?? how to implement - i want to load checkpoints of trained model after specific epochs

from keras.preprocessing.image import ImageDataGenerator

WIDTH = 384
HEIGHT = 216

# Create data generator.
batch_size = 2**6

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
    target_size=(HEIGHT, WIDTH),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
    # save_to_dir="temp"
)

# Modell

input_shape = (82944,)
input_reshape = (HEIGHT, WIDTH, 1)

hidden_num_units = 500
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

nof_epochs = 200

# Callbacks
tensorboard = TensorBoard(log_dir='logs/model-{}epochs-{}batchsize-{}hidden_units'
                          .format(nof_epochs, batch_size, hidden_num_units), histogram_freq=0,
                          write_graph=True, write_images=False,
                          batch_size=batch_size)

# Class Weights

counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: round(max_val/num_images, 2)
                 for class_id, num_images in counter.items()}

print(class_weights)

# Train Model

model.fit_generator(
    train_generator,
    steps_per_epoch=10000//batch_size,
    epochs=nof_epochs,
    callbacks=[tensorboard],
    class_weight=class_weights,
    initial_epoch=0
)

model.save('models/model-{}epochs-{}batchsize-{}hidden_units'.format(nof_epochs, batch_size, hidden_num_units) + '.h5')

# tensorboard --logdir=foo:C:/Users/Mario/PycharmProjects/f1racer/logs
