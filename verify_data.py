import numpy as np
import chess
import filemanager
import tensorflow as tf
import keras
import seaborn
import pandas
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

def get_piece_from_id(id):
    if id == 0:
        return chess.Piece(chess.PAWN, chess.WHITE)
    elif id == 1:
        return chess.Piece(chess.PAWN, chess.BLACK)
    elif id == 2:
        return chess.Piece(chess.KNIGHT, chess.WHITE)
    elif id == 3:
        return chess.Piece(chess.KNIGHT, chess.BLACK)
    elif id == 4:
        return chess.Piece(chess.BISHOP, chess.WHITE)
    elif id == 5:
        return chess.Piece(chess.BISHOP, chess.BLACK)
    elif id == 6:
        return chess.Piece(chess.ROOK, chess.WHITE)
    elif id == 7:
        return chess.Piece(chess.ROOK, chess.BLACK)
    elif id == 8:
        return chess.Piece(chess.QUEEN, chess.WHITE)
    elif id == 9:
        return chess.Piece(chess.QUEEN, chess.BLACK)
    elif id == 10:
        return chess.Piece(chess.KING, chess.WHITE)
    elif id == 11:
        return chess.Piece(chess.KING, chess.BLACK)
    else:
        return None


def get_board_from_bitmap(bitboard, flip):
    board = chess.Board()

    board.clear_board()

    for piece_id in range(12):
        layer = bitboard[piece_id]

        indices = []

        koords = np.transpose(np.where(layer == 1))

        for tuple in koords:
            indices.append(tuple[0] * 8 + tuple[1])

        for index in indices:
            piece = get_piece_from_id(piece_id)
            board.set_piece_at(index, piece)

    if flip:
        board = board.mirror().transform(chess.flip_horizontal)
    return board



if __name__ == '__main__':
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    model = keras.models.load_model('D:\\leguan_data\\leguan_models\\leguan_model.hdf5')

    f1, f2 = filemanager.open_files(2)
    gen = filemanager.batch_generator(100000, f1, f2)
    bitboards, true_labels = next(gen)
    predict_labels = predict_bitboards(bitboards, model)

    #df = pandas.DataFrame({'label': predict_labels})

    conf_matr = confusion_matrix(true_labels, predict_labels)
    display = ConfusionMatrixDisplay(conf_matr).plot()
    #plot = seaborn.countplot(x = 'label', data = df)
    plt.savefig("mygraph.png")

    filemanager.close_files([f1, f2])
