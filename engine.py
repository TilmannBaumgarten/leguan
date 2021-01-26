import tensorflow as tf
from tensorflow import keras
import chess
import numpy as np
from preprocessing import get_bitmap


def evaluate_board(board, model):
    bitboard = get_bitmap(board)

    bitmap = np.reshape(bitboard, (1, 12, 8, 8))

    eval = model.predict((bitmap))

    return eval

if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = keras.models.load_model('D:\\leguan_data\\leguan_models\\leguan_model.hdf5')

    test_position = chess.Board('b2rr3/2pqnpbk/1p4p1/pP1p1PPp/P1PP3P/3QNN2/8/R1BK2R1 b - - 0 1')
    print(test_position)

    evaluation = evaluate_board(test_position, model)
    evaluation = evaluation.argmax(axis=-1)
    print(evaluation)
