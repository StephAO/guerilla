import chess.pgn
from os import listdir
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
    with open(filename, 'r') as pgn:
        game = chess.pgn.read_game(pgn)
        while not game.is_end():
            fen = game.board().fen()
            fen = fen.split(' ')
            if fen[1] == 'b':
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
    return new_fen

def get_fens(num_games):
    ''' Returns a list of fens from games. Will either read from num_games games or all games in folder /pgn_files/single_game_pgns
        inputs:
            num_games:
                number of games to read from
        output:
            fens:
                list of fen strings from games
    '''

    path = '/home/stephane/guerilla/helpers/pgn_files/single_game_pgns'
    files = [f for f in listdir(path)[:num_games] if isfile(join(path, f))]
    fens = []
    for f in files[0:num_games]:
        fens.extend(read_pgn(join(path, f)))
    return fens
# print len(read_pgn('ct-Aronian, Levon-Carlsen, Magnus-2014.2.4__2.pgn'))
