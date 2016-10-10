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
            if fen[1] == 'w':
                fen = dh.flip_board(fen)
            # fen = fen.split(' ')[0]
            fens.append(fen)
            game = game.variation(0)
    return fens


def get_fens(time):
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
    checkpoint_path = dir_path + '/pgn_files/game_num.txt'
    games_path = dir_path + '/pgn_files/single_game_pgns'


    game_num = 0
    if os.path.isfile(checkpoint_path):
        with open(checkpoint_path) as f:
            l = f.readline()
            game_num = int(l)

    files = [f for f in os.listdir(games_path) if isfile(join(games_path, f))]
    
    with open(dir_path + '/../pickles/numbers.nsv', 'a') as f:
        start_time = time.clock()
        while time.clock() - start_time < time:
            game_num += 1
            fens = read_pgn(files[game_num])
            for fen in fens:
                f.write(fen + '\n')




def load_fens(filename='fens.p'):
    """
    Loads the fens pickle.
        Input:
            filename:
                Pickle filename.
        Output:
            Loaded pickle.
    """
    full_path = dir_path + "/../pickles/" + filename
    return pickle.load(open(full_path, 'rb'))


def main():
    retrieve_time = raw_input("How long do you want to generate fens for?:")

    fens = get_fens(time=retrieve_time)


if __name__ == "__main__":
    main()
