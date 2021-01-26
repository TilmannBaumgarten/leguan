from preprocessing import get_bitmap
from verify_data import get_board_from_bitmap
import chess.pgn

pgn = open(f"D:\\leguan_data\\single_game.pgn")
game = chess.pgn.read_game(pgn)

curr_node = game.next()
done = False
while not done:
    if curr_node.is_end():
        done = True
    board = curr_node.board()
    turn = board.turn
    print(board)

    bitboard = get_bitmap(board)
    print(bitboard)
    board = get_board_from_bitmap(bitboard, turn == chess.BLACK)

    print(board)

    curr_node = curr_node.next()

