import tensorflow as tf
from tensorflow import keras
import chess
import numpy as np
from chessengine.preprocessing import get_bitmap
import math

physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
engine = keras.models.load_model('D:\\leguan_data\\leguan_models\\leguan_model_1.0.hdf5')



def predict_bitboards(bitboards, model):
    bitboards = tf.reshape(bitboards, (1,-1,12,8,8))
    eval = model.predict(tuple(bitboards))
    eval = eval.argmax(axis=-1)

    return eval

def evaluate_board(board, model):
    bitboard = get_bitmap(board)

    bitmap = np.reshape(bitboard, (1, 12, 8, 8))

    eval = model.predict((bitmap))
    eval = eval.argmax(axis=-1)

    return eval[0]

def derLeguan(board):
    bitboards = []
    dict = {}
    for move in board.legal_moves:
        board.push(move)
        list = []
        for opp_move in board.legal_moves:
            board.push(opp_move)
            bitboard = get_bitmap(board)
            bitboards.append(bitboard)
            list.append(str(opp_move))
            board.pop()
        board.pop()
        dict[str(move)] = list

    evaluations = predict_bitboards(bitboards, engine)

    """i = 0
    for x in dict:
        print('\n')
        print(x)
        for y in dict[x]:
            print(f'{y} - {evaluations[i]}', end = ', ' )
            i += 1"""

    counter = 0
    best_move = None
    best_reply = None

    best_curr_eval = -math.inf
    best_curr_avg = -math.inf
    for move in dict:

        eval_after_best_reply = math.inf
        sum = 0

        for elem in dict[move]:
            sum = sum + evaluations[counter]

            if evaluations[counter] < eval_after_best_reply:

                best_reply = elem
                eval_after_best_reply = evaluations[counter]

            counter += 1

        avg_eval_after_move = sum/len(dict[move])

        if eval_after_best_reply > best_curr_eval:
            best_curr_avg = avg_eval_after_move
            best_move = move
            best_curr_eval = eval_after_best_reply
        elif eval_after_best_reply == best_curr_eval:
            if avg_eval_after_move > best_curr_avg:
                best_move = move
                best_curr_avg = avg_eval_after_move

    print(best_move)
    print(best_curr_eval)
    return best_move
    """print('\n')
    print('-----------------------')
    print(best_move)
    print(best_reply)
    print(best_curr_eval)
    print(best_curr_avg)"""

"""if __name__ == '__main__':
    board = chess.Board('rnbqk1r1/pp1nbppp/4p3/1BPpP3/5P2/2N2N2/PPP3PP/R1BQK2R b KQq - 0 8')
    derLeguan(board, engine)"""

