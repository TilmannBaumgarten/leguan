import chess.pgn
#import process_games
pgn = open("D:\\leguan_data\\test.pgn")
game = chess.pgn.read_game(pgn)


def label_float(eval, turn):
    eval_string = str(eval.pov(turn))
    if '#-' in eval_string:
        res = -20000
    elif '#' in eval_string:
        res = 20000
    else:
        res = int(eval_string)
    return res

def label_15(eval):
    if eval == -20000:
        res = 0
    elif eval == 20000:
        res = 14
    else:
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


if __name__ == '__main__':
    while game:
        done = False
        curr_node = game.next()
        while not done:
            if curr_node.is_end():
                done = True

            turn = curr_node.turn()
            if turn == chess.BLACK:
                bitmap = process_games.get_bitmap_black(curr_node.board())
            else:
                bitmap = process_games.get_bitmap_white(curr_node.board())

            print(curr_node.board())
            print(curr_node.eval().white())
            print(label_float(curr_node.eval(), curr_node.turn()))
            curr_node = curr_node.next()

        game = chess.pgn.read_game(pgn)
