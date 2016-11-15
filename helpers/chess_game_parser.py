"""
Functions for interacting with single game pgn files.
"""

import sys
import chess.pgn
import pickle
import sys
import os
import time
from os.path import isfile, join

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path + '/../src/')
import data_handler as dh


def read_pgn(filename):
    """
    Given a pgn filename, reads the file and returns a fen for each position in the game.
        Input:
            filename:
                the file nameS
        Output:
            fens:
                list of fen strings
    """
    fens = []
    with open(filename, 'r') as pgn:
        game = chess.pgn.read_game(pgn)
        while not game.is_end():
            fen = game.board().fen()
            if fen.split(' ')[1] == 'b':
                fen = dh.flip_board(fen)
            # fen = fen.split(' ')[0]
            fens.append(fen)
            game = game.variation(0)
    return fens


def get_fens(generate_time):
    """
    Returns a list of fens from games.
    Will either read from num_games games or all games in folder /pgn_files/single_game_pgns.
        Inputs:
            num_games:
                number of games to read from
        Output:
            fens:
                list of fen strings from games
    """
    checkpoint_path = dir_path + '/extracted_data/game_num.txt'
    games_path = dir_path + '/pgn_files/single_game_pgns'

    game_num = 0
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path) as f:
            l = f.readline()
            game_num = int(l)

    files = [f for f in os.listdir(games_path) if isfile(join(games_path, f))]
    
    start_time = time.clock()
    with open(dir_path + '/extracted_data/fens.nsv', 'a') as fen_file:
        print "Opened fens output file..."
        while (time.clock() - start_time) < generate_time:
            fens = read_pgn(games_path + '/' + files[game_num])
            for fen in fens:
                fen_file.write(fen + '\n')

            print "Processed game %d..." % game_num
            game_num += 1

    # Write out next game to be processed
    with open(dir_path + '/extracted_data/game_num.txt', 'w') as num_file:
        num_file.write(str(game_num))

def load_fens(filename='fens.nsv', num_values=None):
    """
    Loads the fens pickle.
        Input:
            filename:
                Pickle filename.
            num_values[int]:
                Max number of stockfish values to return. 
                (will return min of num_values and number of values stored in file)
        Output:
            Loaded pickle.
    """
    full_path = dir_path + "/extracted_data/" + filename
    fens = []
    count = 0
    with open(full_path, 'r') as fen_file:
        for line in fen_file:
            fens.append(line.strip())
            count += 1
            if num_values is not None and count >= num_values:
                break
    print len(fens)
    return fens

def main():

    generate_time = raw_input("How many seconds do you want to generate fens for?: ")

    fens = get_fens(int(generate_time))


if __name__ == "__main__":
    main()
