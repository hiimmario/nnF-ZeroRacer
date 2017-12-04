from collections import Counter

from keras.models import Sequential
from keras.layers import Dense, InputLayer, Conv2D, MaxPooling2D, Flatten,\
                         Dropout, BatchNormalization
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.preprocessing.image import ImageDataGenerator

WIDTH = 192
HEIGHT = 108

# Create data generator.
batch_size = 2**6

train_datagen = ImageDataGenerator(
    rescale=1./255
)

test_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    "images_smaller",
    target_size=(HEIGHT, WIDTH),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True
    # save_to_dir="tmp"
)

test_generator = test_datagen.flow_from_directory(
    "images_smaller_test",
    target_size=(HEIGHT, WIDTH),
    color_mode="grayscale",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
    # save_to_dir="tmp"
)


# test output images
# ctr = 0
# for _ in train_generator:
#     ctr += 1
#     if ctr >= 1:
#         break

# Modell

input_shape = (HEIGHT*WIDTH,)
input_reshape = (HEIGHT, WIDTH, 1)

hidden_num_units = 640
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

    Dropout(0.5),

    Dense(units=hidden_num_units, activation='relu'),

    Dropout(0.3),

    Dense(units=output_num_units, input_dim=hidden_num_units,
          activation='softmax'),
])

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Callbacks
tensorboard = TensorBoard(log_dir='logs/model-{}batchsize-{}hidden_units_10k'
                          .format(batch_size, hidden_num_units), histogram_freq=0,
                          write_graph=True, write_images=False,
                          batch_size=batch_size)

checkpoints = ModelCheckpoint("checkpoints/weights.0_{epoch:02d}-{val_loss:.2f}.hdf5",
                              period=1,
                              save_best_only=True, monitor='val_loss')

# Class Weights

counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: round(max_val/num_images, 2)
                 for class_id, num_images in counter.items()}

print(class_weights)

# Train Model

nof_epochs = 16

model.fit_generator(
    train_generator,
    steps_per_epoch=20000//batch_size,
    epochs=nof_epochs,
    validation_data=test_generator,
    validation_steps=2172//batch_size,
    callbacks=[tensorboard, checkpoints],
    class_weight=class_weights,
    initial_epoch=4
)

model.save('models/model-{}batchsize-{}hidden_units'.format(batch_size, hidden_num_units) + '.h5')



# tensorboard --logdir=foo:C:/Users/Mario/PycharmProjects/f1racer/logs