import tensorflow as tf
from tensorflow.keras import datasets
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization
from keras.optimizers import SGD
import numpy as np

CLASSES = 10
EPOCHS = 100
BATCH_SIZE = 128
OPT = SGD(lr=0.01)
ACTIVATION = "elu"
LOSS_FUNCTION = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def model():
    model = Sequential()
    model.add(Convolution2D(20, (5, 5), input_shape=(32, 32, 3), padding='same', activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Convolution2D(50, (3, 3), padding='same', activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(250, activation=ACTIVATION))
    model.add(Dense(CLASSES))
    model.add(Activation('softmax'))

    model.compile(loss=LOSS_FUNCTION, optimizer=OPT, metrics=['accuracy'])

    model.summary()

    return model


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    (train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()
    train_images, test_images = train_images / 255.0, test_images / 255.0

    print(type(train_images))

    model = model()

    history = model.fit(train_images, train_labels, validation_data=(test_images, test_labels), epochs=10)

    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

