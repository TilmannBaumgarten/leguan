import numpy as np
import tables


def init(pid):
    test = np.zeros(shape=(12, 8, 8))
    pos_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_positions_{pid}.hdf5", mode='w')
    eval_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_evals_{pid}.hdf5", mode='w')
    atom = tables.Atom.from_dtype(test.dtype)
    pos_file.create_earray(pos_file.root, 'data', atom, (0, 12, 8, 8))
    eval_file.create_earray(eval_file.root, 'data', atom, (0,))
    pos_file.close()
    eval_file.close()


def write(position, label, pid):
    pos_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_positions_{pid}.hdf5", mode='a')
    eval_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_evals_{pid}.hdf5", mode='a')

    pos_file.root.data.append([position])
    eval_file.root.data.append([label])

    pos_file.close()
    eval_file.close()


def open_file(id):
    positions = []
    evaluations = []

    pos_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_positions_{id}.hdf5", mode='r')
    eval_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_evals_{id}.hdf5", mode='r')

    for x in range(len(pos_file.root.data)):
        positions.append(pos_file.root.data[x])
        evaluations.append(eval_file.root.data[x])

    pos_file.close()
    eval_file.close()

    return [positions, evaluations]


if __name__ == '__main__':
    init(0)
