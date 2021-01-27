from preprocessing import get_bitmap
from verify_data import get_board_from_bitmap
import chess.pgn
import filemanager
import tables

import numpy as np

def onehot(label):
    arr = np.zeros(15)
    arr[label] = 1
    return arr

