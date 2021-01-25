import tables
import numpy as np
import chess


def get_piece_from_id(id):
    if id == 0:
        return chess.Piece(chess.BISHOP, chess.BLACK)
    elif id == 1:
        return chess.Piece(chess.BISHOP, chess.WHITE)
    elif id == 2:
        return chess.Piece(chess.PAWN, chess.BLACK)
    elif id == 3:
        return chess.Piece(chess.PAWN, chess.WHITE)
    elif id == 4:
        return chess.Piece(chess.QUEEN, chess.BLACK)
    elif id == 5:
        return chess.Piece(chess.QUEEN, chess.WHITE)
    elif id == 6:
        return chess.Piece(chess.KING, chess.BLACK)
    elif id == 7:
        return chess.Piece(chess.KING, chess.WHITE)
    elif id == 8:
        return chess.Piece(chess.KNIGHT, chess.BLACK)
    elif id == 9:
        return chess.Piece(chess.KNIGHT, chess.WHITE)
    elif id == 10:
        return chess.Piece(chess.ROOK, chess.BLACK)
    elif id == 11:
        return chess.Piece(chess.ROOK, chess.WHITE)
    else:
        return None


def get_pos_from_8x8(matrix):
    koords = zip(*np.nonzero(matrix))
    res = []
    for item in koords:
        res.append(63 - (8 * item[0] + 7 - item[1]))
    return res


def verify_data_manually():
    pos_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_positions_0.hdf5", mode='r')
    eval_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_evals_0.hdf5", mode='r')

    test_batch = pos_file.root.data[:100]
    test_batch_eval = eval_file.root.data[:100]

    pos_file.close()
    eval_file.close()

    board = chess.Board()

    for x in range(100):

        bitboard = test_batch[x]
        evaluation = test_batch_eval[x]

        board.clear_board()

        for y in range(12):
            nums = get_pos_from_8x8(bitboard[y])
            for item in nums:
                piece = get_piece_from_id(y)
                board.set_piece_at(item, piece)

        print(board)
        print(evaluation)


if __name__ == '__main__':
    verify_data_manually()
