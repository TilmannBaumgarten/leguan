import chess.pgn
import multiprocessing
from multiprocessing import Process
import process_games
import filemanager
import time


def worker(queue, pid):
    pgn = open(f"D:\\leguan_data\\use\\{pid}.pgn")
    game = chess.pgn.read_game(pgn)
    while game:
        positions = process_games.process_game(game)
        for pos in positions:
            queue.put(pos)

        game = chess.pgn.read_game(pgn)


def listener(queue, pid):
    while True:
        [bitmap, label] = queue.get()
        filemanager.write(bitmap, label, pid)


if __name__ == '__main__':
    queue = multiprocessing.Queue(maxsize=10000)

    for x in range(6):
        filemanager.init(x)

    p0 = Process(target=worker, args=(queue, 0))
    p1 = Process(target=worker, args=(queue, 1))
    p2 = Process(target=worker, args=(queue, 2))

    p3 = Process(target=listener, args=(queue, 0))
    p4 = Process(target=listener, args=(queue, 1))
    p5 = Process(target=listener, args=(queue, 2))
    p6 = Process(target=listener, args=(queue, 3))
    p7 = Process(target=listener, args=(queue, 4))
    p8 = Process(target=listener, args=(queue, 5))

    p0.start()
    p1.start()
    p2.start()
    p3.start()
    p4.start()
    p5.start()
    p6.start()
    p7.start()
    p8.start()

    p0.join()
    p1.join()
    p2.join()

    while not queue.empty():
        time.sleep(60)

    p2.kill()
    p3.kill()
    p4.kill()
    p5.kill()
    p6.kill()
    p7.kill()
    p8.kill()
