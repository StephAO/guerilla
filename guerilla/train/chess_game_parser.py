"""
Functions for interacting with single game pgn files.
"""

import os
import time
from os.path import isfile, join
import random as rnd
import numpy as np

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


def get_fens(generate_time, num_random=0, store_prob=0.0):
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
    fen_count = 0
    fname = 'fens_sampled'
    print "starting at game num: {}".format(game_num)
    with open(resource_filename('guerilla', 'data/extracted_data/{}.csv'.format(fname)), 'w') as fen_file:
        print "Opened fens output file..."
        while (time.clock() - start_time) < generate_time and game_num < len(files):
            fens = read_pgn(games_path + '/' + files[game_num], max_skip=0)

            # Randomly choose 3
            fens = np.random.choice(fens, 3, replace=False)

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
                    fen_count += 1
            if game_num % 100 == 0:
                print "Processed game %d [%d] fens..." % (game_num, fen_count)
            game_num += 1

    # Write out next game to be processed
    with open(resource_filename('guerilla', 'data/extracted_data/game_num.txt'), 'w') as num_file:
        num_file.write(str(game_num))

def iter_pgncollection(pgn_col):
    """
    Iterator for games in the provided PGN collection. A PGN collection is a single file with multiple PGNs.
    :param pgn_col: [File Object] File object for PGN collection. (i.e. already opened file)
    :return: [chess.pgn.Game] yields chess games.
    """
    new_pgn_key = '[Event'
    temp_file = 'temp.pgn'

    # Iterate through lines of pgn collection
    pgn_lines = []
    for line in pgn_col:
        # We've reached a new pgn!
        if line.split(' ')[0] == new_pgn_key:
            # If we just completed reading a pgn, write PGN to file, read PGN and yield game
            if pgn_lines:
                with open(temp_file, 'w') as f:
                    f.writelines(pgn_lines)
                with open(temp_file, 'r') as f:
                    game = chess.pgn.read_game(f)
                yield game
                # Reset PGN buffer
                pgn_lines = []

        # Add to PGN buffer
        pgn_lines.append(line)

    # Remove tempfile
    os.remove(temp_file)

def get_checkmate_fens():
    """
    Extracts checkmate and pre-mate FEN files from PGN collections.
    Reads all PGN collections in pgn_files/kingbase/.
    Can directly read KingBase files. Download from http://www.kingbase-chess.net/ then extract into kingbase directory.
    """
    db_path = resource_filename('guerilla', 'data/pgn_files/kingbase')

    game_count = mate_count = 0
    pgn_collections = [join(db_path, f) for f in os.listdir(db_path) if join(db_path, f)]

    with open(resource_filename('guerilla', 'data/extracted_data/checkmate_fens_temp.csv'), 'w') as mate_file, \
         open(resource_filename('guerilla', 'data/extracted_data/premate_fens_temp.csv'), 'w') as pre_file:

        print "Opened checkmate and premate fens output file..."

        for f in pgn_collections:
            print "Reading through collection {}...".format(f)
            with open(f, 'r') as pgn_col:
                for game in iter_pgncollection(pgn_col):
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

                    game_count += 1
                    if game_count % 1000 == 0:
                        print "%d %d" % (game_count, mate_count)


def load_fens(filename='fens.csv', num_values=None):
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

    get_fens(int(generate_time), num_random=4)


if __name__ == "__main__":
    main()
