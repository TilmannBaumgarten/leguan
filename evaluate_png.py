import chess.pgn
import chess.engine

engine = chess.engine.SimpleEngine.popen_uci('C:\\Users\\Tilmann\\Desktop\\stockfish_12_win_x64_avx2\\stockfish_20090216_x64_avx2.exe')
pgn = open("D:\\lichess games\\single_game.pgn")


def analyseGame(game):
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        print(board)
        print(info['score'])

if __name__ == '__main__':
    game = chess.pgn.read_game(pgn)
    analyseGame(game)

engine.quit()