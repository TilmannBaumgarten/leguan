import chess.pgn
import chess
import multiprocessing
from multiprocessing import Process
import numpy as np
import time
import tables

PieceList = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def get_label(eval, turn):
    eval = str(eval.pov(turn))
    if '#-' in eval:
        res = 0
    elif '#' in eval:
        res = 14
    else:
        eval = int(eval)
        if eval >= 30:
            if eval >= 400:
                if eval >= 1500:
                    res = 13
                else:
                    if eval >= 700:
                        res = 12
                    else:
                        res = 11
            else:
                if eval >= 150:
                    res = 10
                else:
                    if eval >= 70:
                        res = 9
                    else:
                        res = 8
        else:
            if eval >= -150:
                if eval >= -30:
                    res = 7
                else:
                    if eval >= -70:
                        res = 6
                    else:
                        res = 5

            else:
                if eval >= -700:
                    if eval >= -400:
                        res = 4
                    else:
                        res = 3
                else:
                    if eval >= -1500:
                        res = 2
                    else:
                        res = 1
    return res


def create_files():
    atom = tables.Atom.from_dtype(np.dtype('float16'))

    for x in range(6):
        pos_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_positions_{x}.hdf5", mode='w')
        eval_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_evals_{x}.hdf5", mode='w')

        pos_file.create_earray(pos_file.root, 'data', atom, (0, 12, 8, 8))
        eval_file.create_earray(eval_file.root, 'data', atom, (0,))

        pos_file.close()
        eval_file.close()


def to_file(bitboard, label, pid):
    pos_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_positions_{pid}.hdf5", mode='a')
    eval_file = tables.open_file(f"D:\\leguan_data\\training_data\\train_evals_{pid}.hdf5", mode='a')

    pos_file.root.data.append([bitboard])
    eval_file.root.data.append([label])

    pos_file.close()
    eval_file.close()


def get_bitmap(board):
    turn = board.turn

    if turn == chess.BLACK:
        board = board.mirror().transform(chess.flip_horizontal)

    bitmap = []

    for piece in PieceList:
        for color in [chess.WHITE, chess.BLACK]:
            layer = board.pieces(piece, color).tolist()
            layer = np.reshape(layer, (8, 8))
            layer = layer * 1
            bitmap.append(layer)

    return bitmap


def process_game(game):
    processed_game = []

    curr_node = game.next()
    done = False
    while not done:
        if curr_node.is_end():
            done = True

        board = curr_node.board()

        bitmap = get_bitmap(board)

        if curr_node.eval():
            eval = curr_node.eval()

        label = get_label(eval, board.turn)

        tuple = [bitmap, label]
        processed_game.append(tuple)

        curr_node = curr_node.next()

    return processed_game


def worker(queue, pid):
    pgn = open(f"D:\\leguan_data\\use\\{pid}.pgn")
    game = chess.pgn.read_game(pgn)
    while game:
        processed_game = process_game(game)
        for tuple in processed_game:
            queue.put(tuple)

        game = chess.pgn.read_game(pgn)


def listener(queue, pid):
    while True:
        [bitboard, label] = queue.get()
        to_file(bitboard, label, pid)


if __name__ == '__main__':
    queue = multiprocessing.Queue(maxsize=10000)

    create_files()

    workers = [
        Process(target=worker, args=(queue, 0)),
        Process(target=worker, args=(queue, 1)),
        Process(target=worker, args=(queue, 2))
    ]

    listeners = [
        Process(target=listener, args=(queue, 0)),
        Process(target=listener, args=(queue, 1)),
        Process(target=listener, args=(queue, 2)),
        Process(target=listener, args=(queue, 3)),
        Process(target=listener, args=(queue, 4)),
        Process(target=listener, args=(queue, 5))
    ]

    for w in workers:
        w.start()

    for l in listeners:
        l.start()

    for w in workers:
        w.join()

    while not queue.empty():
        time.sleep(60)

    for l in listeners:
        l.kill()
