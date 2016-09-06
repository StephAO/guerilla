import sys
import chess.pgn
import pickle
import sys
import os
from os.path import isfile, join

def read_pgn(filename):
    ''' given a pgn filename, reads the file and returns a fen for each position in the game
        input:
            filename:
                the file nameS
        output:
            fens:
                list of fen strings
    '''
    fens = []
    i = 0
    with open(filename, 'r') as pgn:
        game = chess.pgn.read_game(pgn)
        while not game.is_end():
            fen = game.board().fen()
            fen = fen.split(' ')
            if fen[1] == 'w':
                fen = flip_board(fen[0])
            else:
                fen = fen[0]
            fens.append(fen)
            game = game.variation(0)
    return fens

def flip_board(fen):
    ''' switch colors of pieces
        input:
            fen:
                fen string (only board state)
        output:
            new_fen:
                fen string with colors switched
    '''
    new_fen = ''
    for char in fen:
        if char.isupper():
            new_fen += char.lower()
        elif char.islower():
            new_fen += char.upper()
        else:
            new_fen += char
    new_fen_list = new_fen.split('/')
    new_fen = '/'.join(new_fen_list[::-1])
    return new_fen

def get_fens(num_games=-1):
    ''' Returns a list of fens from games. Will either read from num_games games or all games in folder /pgn_files/single_game_pgns
        inputs:
            num_games:
                number of games to read from
        output:
            fens:
                list of fen strings from games
    '''
    # @TODO fix this shitty file path
    path = dir_path + '/pgn_files/single_game_pgns'
    files = [f for f in os.listdir(path)[:num_games] if isfile(join(path, f))]
    fens = []
    for f in files[:num_games]:
        fens.extend(read_pgn(join(path, f)))
    
    return fens

def load_fens(filename='fens.p'):
    full_path = dir_path + "/../pickles/" + filename
    return pickle.load(open(full_path, 'rb'))

def main():
    number_of_games = -1
    if len(sys.argv) > 1:
        number_of_games = int(sys.argv[1])
    
    fens = get_fens(num_games=number_of_games)
    
    if len(sys.argv) > 2:
        number_of_fens = int(sys.argv[2])
        fens = fens[:number_of_fens]

    pickle_path = dir_path + '/../pickles/fens.p'
    pickle.dump(fens, open(pickle_path, 'wb'))

dir_path = os.path.dirname(__file__)
if __name__ == "__main__":
    main()

    