import tables
import tensorflow as tf
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization

CLASSES = 15
EPOCHS = 10
BATCH_SIZE = 128
OPT = SGD(lr=0.01)
ACTIVATION = "elu"
LOSS_FUNCTION = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
DATA_FILE_ID = 0
VALIDATION_DATA_FILE_ID = 1


def get_model():
    model = Sequential()
    model.add(Convolution2D(100, (8, 8), input_shape=(12, 8, 8), data_format='channels_first', padding='valid',
                            activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Convolution2D(50, (5, 5), data_format='channels_first', padding='valid', activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Convolution2D(50, (5, 5), data_format='channels_first', padding='valid', activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(250, activation=ACTIVATION))
    model.add(Dense(CLASSES))
    model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUNCTION, optimizer=OPT, metrics=['accuracy'])
    return model


def batch_generator(batch_size, file_id):
    pos_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_positions_{file_id}.hdf5", mode='r')
    eval_file = tables.open_file(f"D:\\leguan_data\\training_data\\new_evals_{file_id}.hdf5", mode='r')

    len = get_file_length(file_id)
    i = 0
    while True:
        try:
            data = pos_file.root.data[i * batch_size:(i + 1) * batch_size]
            labels = eval_file.root.test[i * batch_size:(i + 1) * batch_size]
        except tables.exceptions.NoSuchNodeError:
            return

        i += 1
        if i == int(len / batch_size):
            i = 0
        res = (data, labels)
        yield res

    pos_file.close()
    eval_file.close()


def get_file_length(id):
    eval_file = tables.open_file(f"D:\\leguan_data\\training_data\\new_evals_{id}.hdf5", mode='r')
    len = len(eval_file.root.data)
    eval_file.close()
    return len


def main():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    gen = batch_generator(BATCH_SIZE, DATA_FILE_ID)
    valid_gen = batch_generator(BATCH_SIZE, VALIDATION_DATA_FILE_ID)

    steps = int(get_file_length(DATA_FILE_ID) / BATCH_SIZE)
    validation_steps = int(steps / 10)

    model = get_model()

    model.summary()

    model.fit_generator(gen, epochs=EPOCHS, verbose=1, shuffle=True, steps_per_epoch=steps, validation_data=valid_gen,
                        validation_steps=validation_steps)

    model.save('D:\\leguan_data\\leguan_models\\leguan_model.hdf5')


if __name__ == '__main__':
    main()
