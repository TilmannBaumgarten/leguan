import tensorflow as tf
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization, Input
import filemanager

EPOCHS = 10
BATCH_SIZE = 128
OPT = SGD(lr=0.01)
ACTIVATION = "relu"
LOSS_FUNCTION = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


def get_model():
    model = Sequential()
    model.add(Convolution2D(128, (8, 8), input_shape=(12, 8, 8), data_format='channels_first', padding='same', activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Convolution2D(64, (5, 5), padding='same', data_format='channels_first', activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Convolution2D(32, (3, 3), padding='same', data_format='channels_first', activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(250, activation=ACTIVATION))
    model.add(Dense(15))
    model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUNCTION, optimizer=OPT, metrics=['accuracy'])
    return model


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    pos_file, eval_file = filemanager.open_files(0)
    pos_file_valid, eval_file_valid = filemanager.open_files(1)

    gen = filemanager.batch_generator(BATCH_SIZE, pos_file, eval_file)
    valid_gen = filemanager.batch_generator(BATCH_SIZE, pos_file_valid, eval_file_valid)

    steps = int(filemanager.get_file_length(eval_file) / BATCH_SIZE)
    validation_steps = int(steps / 10)

    model = get_model()

    model.summary()

    model.fit_generator(gen, epochs=EPOCHS, verbose=1, shuffle=True, steps_per_epoch=steps, validation_data=valid_gen, validation_steps=validation_steps)

    model.save('D:\\leguan_data\\leguan_models\\leguan_model_1.2.hdf5')

    filemanager.close_files([pos_file, eval_file, pos_file_valid, eval_file_valid])


if __name__ == '__main__':
    main()
