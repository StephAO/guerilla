"""
Functions for interacting with single game pgn files.
"""

import os
import time
from os.path import isfile, join
import random as rnd
import pickle

import chess.pgn
from pkg_resources import resource_filename

import guerilla.data_handler as dh


def read_pgn(filename, max_skip=80):
    """
    Given a pgn filename, reads the file and returns a fen for each position in the game.
        Input:
            filename [String]:
                the file name
            max_skip [Int]:
                The maximum number of half-moves which are skipped.
        Output:
            fens:
                list of fen strings
    """
    fens = []
    with open(filename, 'r') as pgn:
        game = chess.pgn.read_game(pgn)
        while True:
            fens.append(game.board().fen())
            if game.is_end():
                break
            game = game.variation(0)

    # Down sample based on half-move count
    max_skip = min(max_skip, len(fens) - 1)
    skip_start = rnd.randint(0, max_skip)

    return fens[skip_start:]


def get_fens(generate_time, num_random=2, store_prob=0.0):
    """
    Returns a list of fens from games.
    Will either read from all games in folder /pgn_files/single_game_pgns.
        Inputs:
            num_games:
                number of games to read from
        Output:
            fens:
                list of fen strings from games
    """

    # Set seed so that results are reproducable
    rnd.seed(123456)

    checkpoint_path = resource_filename('guerilla', 'data/extracted_data/game_num.txt')
    games_path = resource_filename('guerilla', 'data/pgn_files/single_game_pgns')

    game_num = 0
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path) as f:
            l = f.readline()
            game_num = int(l)

    files = [f for f in os.listdir(games_path) if isfile(join(games_path, f))]

    start_time = time.clock()
    with open(resource_filename('guerilla', 'data/extracted_data/fens.nsv'), 'a') as fen_file:
        print "Opened fens output file..."
        while (time.clock() - start_time) < generate_time:
            fens = read_pgn(games_path + '/' + files[game_num])
            for fen in fens:

                board = chess.Board(fen)

                # When using random moves, store original board with some probability
                out_fens = []
                if num_random > 0 and rnd.random() < store_prob:
                    out_fens.append(fen)

                # Default: Make EACH PLAYER do a random move and then store
                for i in range(num_random):
                    if not list(board.legal_moves):
                        break
                    board.push(rnd.choice(list(board.legal_moves)))

                else:
                    # only store if all random moves were applied
                    out_fens.append(board.fen())

                for out_fen in out_fens:
                    # flip board if necessary
                    if dh.black_is_next(out_fen):
                        out_fen = dh.flip_board(out_fen)

                    fen_file.write(out_fen + '\n')

            print "Processed game %d..." % game_num
            game_num += 1

    # Write out next game to be processed
    with open(resource_filename('guerilla', 'data/extracted_data/game_num.txt'), 'w') as num_file:
        num_file.write(str(game_num))


def load_fens(filename='fens.nsv', num_values=None):
    """
    Loads the fens file.
        Input:
            filename:
                Pickle filename.
            num_values[int]:
                Max number of stockfish values to return. 
                (will return min of num_values and number of values stored in file)
        Output:
        fens [List]
            Loaded fens
    """
    full_path = resource_filename('guerilla', 'data/extracted_data/' + filename)
    fens = []
    count = 0
    with open(full_path, 'r') as fen_file:
        for line in fen_file:
            fens.append(line.strip())
            count += 1
            if num_values is not None and count >= num_values:
                break
    return fens


def main():
    generate_time = raw_input("How many seconds do you want to generate fens for?: ")

    get_fens(int(generate_time))

if __name__ == "__main__":
    main()
