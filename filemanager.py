import tables
import numpy as np
from preprocessing import sigmoid

def batch_generator(batch_size, pos_file, eval_file):
    length = get_file_length(eval_file)
    i = 0
    while True:
        try:
            data = pos_file.root.data[i * batch_size:(i + 1) * batch_size]
            labels = eval_file.root.data[i * batch_size:(i + 1) * batch_size]
        except tables.exceptions.NoSuchNodeError:
            return




        #sigmoid_v = np.vectorize(sigmoid)
        #labels = sigmoid_v(labels)

        i += 1
        if i == int(length / batch_size):
            i = 0
        res = (data, labels)
        yield res


def get_file_length(file):
    length = len(file.root.data)
    return length


def open_files(file_id):
    pos_file = tables.open_file(f"D:\\leguan_data\\float_training_data\\train_positions_{file_id}.hdf5", mode='r')
    eval_file = tables.open_file(f"D:\\leguan_data\\float_training_data\\train_evals_{file_id}.hdf5", mode='r')

    return pos_file, eval_file


def close_files(filelist):
    for file in filelist:
        file.close()
