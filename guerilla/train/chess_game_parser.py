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


def read_pgn(filename, num_skip=0):
    """
    Given a pgn filename, reads the file and returns a fen for each position in the game.
        Input:
            filename:
                the file nameS
            min_move:
                Number of moves which are skipped and for which no FEN is stored.
        Output:
            fens:
                list of fen strings
    """
    fens = []
    move_count = 0
    with open(filename, 'r') as pgn:
        game = chess.pgn.read_game(pgn)
        while not game.is_end():
            # Don't store the first 'num_skip' moves.
            if move_count >= num_skip:
                fen = game.board().fen()
                fens.append(fen)
            game = game.variation(0)
            move_count += 1
    return fens


def num_skip_func(goal_num, curr_num):
    """
    Function defining the number of moves to skip in each game based on the goal and current number of fens.
    Input:
        goal_num [Int]
            Goal number of fens to generate.
        curr_num [Int]
            Number of fens generated thus far.

    Output:
        skip [Int]
            Number of fens to skip in the next game processed.
    """

    ALL_PERC = 3
    NO_START_PERC = 30
    MAX_SKIP = 40  # Note: On average 40 moves per game

    curr_percent = float(curr_num) * 100 / goal_num

    # Store everything (starting board included) for first 5% of fens
    if curr_percent < ALL_PERC:
        return 0

    # Store everything but starting board up until 50% of fens
    if curr_percent < NO_START_PERC:
        return 1

    # Skip more and more fens as you progress
    skip = (MAX_SKIP) * float(curr_percent - NO_START_PERC) / (100 - NO_START_PERC)

    return int(skip)

    # TODO: Shuffle validation fens


def get_fens(generate_time, goal_num=100000, num_random=0, store_prob=0.25):
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
    fen_count = 0
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path) as f:
            data = pickle.load(f)
            game_num = data['game_num']
            print "Goal number of %d was overwritten by stored goal number of %d" % (goal_num, data['goal_num'])
            goal_num = data['goal_num']
            fen_count = data['fen_count']

    files = [f for f in os.listdir(games_path) if isfile(join(games_path, f))]

    start_time = time.clock()
    with open(resource_filename('guerilla', 'data/extracted_data/fens.nsv'), 'a') as fen_file:
        print "Opened fens output file..."
        while (time.clock() - start_time) < generate_time:
            fens = read_pgn(games_path + '/' + files[game_num], num_skip = num_skip_func(goal_num, fen_count))
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
                    fen_count += 1

            print "Processed game %d..." % game_num
            game_num += 1

            if fen_count > goal_num:
                print "Goal number of fens (%d) reached!" % goal_num
                break

    # Write out next game to be processed
    with open(resource_filename('guerilla', 'data/extracted_data/cgp_data.p'), 'w') as cgp_file:
        pickle.dump({'game_num': game_num,
                     'goal_num': goal_num,
                     'fen_count': fen_count}, cgp_file)

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

    fens = get_fens(int(generate_time), goal_num=10000)

    # white_win = black_win = draw = 0
    #
    # games_path = resource_filename('guerilla', 'data/pgn_files/single_game_pgns')
    # files = [f for f in os.listdir(games_path) if isfile(join(games_path, f))]
    # for i in range(1264):
    #     filename = games_path + '/' + files[i]
    #     with open(filename, 'r') as pgn:
    #         game = chess.pgn.read_game(pgn)
    #         if game.headers['Result'] == '0-1':
    #             black_win += 1
    #         elif game.headers['Result'] == '1-0':
    #             white_win += 1
    #         else:
    #             draw += 1
    #
    # print "White win " + str(white_win)
    # print "Black win " + str(black_win)
    # print "Draw " + str(draw)


if __name__ == "__main__":
    main()
