import numpy as np
import chess
from network import batch_generator


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
    gen = batch_generator(1, 0)
    for x in range(5):
        bitboard, eval = next(gen)[0]
        board = get_board_from_bitmap(bitboard, False)
        print(board)
        print(eval)
