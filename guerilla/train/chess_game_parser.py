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
    Extracts fens from games and saves them.
    Will read from all games in folder /pgn_files/single_game_pgns.
        Inputs:
            num_games:
                Amount of time to generate games for.
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
                    out_fen = dh.flip_to_white(out_fen)

                    fen_file.write(out_fen + '\n')

            print "Processed game %d..." % game_num
            game_num += 1

    # Write out next game to be processed
    with open(resource_filename('guerilla', 'data/extracted_data/game_num.txt'), 'w') as num_file:
        num_file.write(str(game_num))


def get_checkmate_fens():
    """
    Extracts checkmate and pre-mate FEN files from games. Saves them to separate files.
    Will read from all games in folder /pgn_files/single_game_pgns.
    """
    games_path = resource_filename('guerilla', 'data/pgn_files/single_game_pgns')

    mate_count = 0
    with open(resource_filename('guerilla', 'data/extracted_data/checkmate_fens.csv'), 'w') as mate_file, \
            open(resource_filename('guerilla', 'data/extracted_data/premate_fens.csv'), 'w') as pre_file:
        print "Opened checkmate and premate fens output file..."
        for i, file in enumerate(os.listdir(games_path)):
            if not os.path.isfile(join(games_path, file)):
                continue
            with open(games_path + '/' + file, 'r') as pgn:
                game = chess.pgn.read_game(pgn)
                result = game.headers['Result']
                if result != '1/2-1/2':
                    # Game was not a draw
                    last_board = game.end().board()
                    pre_board = game.end().parent.board()

                    if last_board.is_checkmate():
                        if result == '1-0':
                            # White checkmated black
                            mate_file.write(dh.flip_board(last_board.fen()) + '\n')
                            pre_file.write(pre_board.fen() + '\n')
                        else:
                            # Black checkmated white
                            mate_file.write(last_board.fen() + '\n')
                            pre_file.write(dh.flip_board(pre_board.fen()) + '\n')

                        mate_count += 1

            if i % 10000 == 0:
                print "%d %d" % (i, mate_count)


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

    if num_values > count:
        raise ValueError(
            "Could not load desired number of fens! File %s only has %d FENs and requested load was %d FENs" % (
                filename, count, num_values))
    return fens


def main():
    generate_time = raw_input("How many seconds do you want to generate fens for?: ")

    get_fens(int(generate_time))


if __name__ == "__main__":
    main()
