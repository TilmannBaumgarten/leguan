import tables
import tensorflow as tf
from keras.optimizers import SGD, Adam
from keras.models import Sequential, Model
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense, Dropout
from keras.layers import BatchNormalization
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from labeling import label_15

CLASSES = 15
EPOCHS = 100
BATCH_SIZE = 128
OPT = SGD(lr=0.01)
ACTIVATION = "elu"
LOSS_FUNCTION = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) #'mse'
LABELING = 'CATEGORICAL' #'float'

def get_model():
    model = Sequential()
    model.add(Convolution2D(20, (5, 5), input_shape=(12, 8, 8), data_format='channels_first', padding='same', activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Convolution2D(50, (3, 3), data_format='channels_first', padding='same', activation=ACTIVATION))
    model.add(BatchNormalization())
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(250, activation=ACTIVATION))
    model.add(Dense(CLASSES))
    model.add(Activation('softmax'))
    model.compile(loss=LOSS_FUNCTION, optimizer=OPT, metrics=['accuracy'])
    return model


def get_data(low, high, percent):
    pos_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_positions_0.hdf5", mode='r')
    eval_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_evals_0.hdf5", mode='r')

    train_data = pos_file.root.data[low:(high*percent)]
    validation_data = pos_file.root.data[(high*percent):high]
    train_labels = eval_file.root.data[low:(high*percent)]
    validation_labels = eval_file.root.data[(high*percent):high]

    if(LABELING == 'float'):
        train_labels = train_labels.reshape(-1, 1)
        validation_labels = validation_labels.reshape(-1, 1)
        scaler = MinMaxScaler(-1,1)
        scaler.fit(train_labels)
        scaler.transform(train_labels)
        scaler.transform(validation_labels)
    elif(LABELING == 'CATEGORICAL'):
        vec = np.vectorize(label_15)

        train_labels = vec(train_labels)
        validation_labels = vec(validation_labels)


    train_tensors = tf.convert_to_tensor(train_data, dtype=tf.float16)
    train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float16)

    train_dataset = tf.data.Dataset.from_tensor_slices((train_tensors, train_labels))
    train_dataset = train_dataset.shuffle(10000)
    train_dataset = train_dataset.batch(BATCH_SIZE)

    validation_tensors = tf.convert_to_tensor(validation_data, dtype=tf.float16)
    validation_labels = tf.convert_to_tensor(validation_labels, dtype=tf.float16)

    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_tensors, validation_labels))
    validation_dataset = validation_dataset.shuffle(10000)
    validation_dataset = validation_dataset.batch(BATCH_SIZE)

    pos_file.close()
    eval_file.close()

    return train_dataset, validation_dataset


if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    train_data, validation_data = get_data(0, 100000, 0.9)

    print(train_data)
    print(validation_data)
    model = get_model()

    model.summary()
    model.fit(train_data, epochs=EPOCHS, verbose=1, shuffle=True, validation_data=validation_data)
    model.save('D:\\leguan_data\\leguan_models\\leguan_model.hdf5')
