import tables
import numpy as np
from sklearn.preprocessing import MinMaxScaler
ID = 0

if __name__ == '__main__':
    atom = tables.Atom.from_dtype(np.dtype('float16'))

    old_eval_file = tables.open_file(f"D:\\leguan_data\\float_training_data\\train_evals_{ID}.hdf5", mode='r')

    new_eval_file = tables.open_file(f"D:\\leguan_data\\float_training_data\\normalized_train_evals_{ID}.hdf5", mode='w')

    new_eval_file.create_earray(new_eval_file.root, 'data', atom, (0,))

    new_eval_file = tables.open_file(f"D:\\leguan_data\\float_training_data\\normalized_train_evals_{ID}.hdf5", mode='a')

    array = old_eval_file.root.data
    array = np.clip(array, -50, 50)

    array = np.reshape(array, (-1, 1))

    scaler = MinMaxScaler()

    scaler.fit(array)

    new_array = scaler.transform(array)

    for item in new_array:
        print(item)
        new_eval_file.root.data.append(item)

