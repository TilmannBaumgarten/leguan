import chess.pgn
import numpy as np
from labeling import label_float
import filemanager

pgn = open("D:\\leguan_data\\test.pgn")
PieceList = [chess.BISHOP, chess.PAWN, chess.QUEEN, chess.KING, chess.KNIGHT, chess.ROOK]


def process_game(game):
    positions = []

    curr_node = game.next()
    done = False
    while not done:
        if curr_node.is_end():
            done = True

        turn = curr_node.turn()

        if turn == chess.BLACK:
            bitmap = get_bitmap_black(curr_node.board())
        else:
            bitmap = get_bitmap_white(curr_node.board())

        if curr_node.eval():
            eval = curr_node.eval()

        y = label_float(eval, turn)

        item = [bitmap, y]
        positions.append(item)

        curr_node = curr_node.next()

    return positions


def get_bitmap_black(board):
    bitmap = []
    for piece in PieceList:
        for color in [chess.WHITE, chess.BLACK]:
            koords = []
            for place in board.pieces(piece, color):
                koords.append(place)

            tuples = []
            for koord in koords:
                tuples.append([int(koord / 8), koord % 8])
            tuples = np.array(tuples)

            piece_bitmap = np.zeros((8, 8))
            if len(tuples) > 0:
                piece_bitmap[tuple(tuples.T)] = 1

            bitmap.append(piece_bitmap)

    return bitmap

def get_bitmap_white(board):
    bitmap = []
    for piece in PieceList:
        for color in [chess.BLACK, chess.WHITE]:
            koords = []
            for place in board.pieces(piece, color):
                koords.append(place)

            tuples = []
            for koord in koords:
                tuples.append([7 - int(koord / 8), koord % 8])
            tuples = np.array(tuples)

            piece_bitmap = np.zeros((8, 8))
            if len(tuples) > 0:
                piece_bitmap[tuple(tuples.T)] = 1

            bitmap.append(piece_bitmap)

    return bitmap


if __name__ == '__main__':
    filemanager.init()
    game = chess.pgn.read_game(pgn)
    while game:
        process_game(game)
        game = chess.pgn.read_game(pgn)
    filemanager.read_print_files()