import chess.pgn

def read_pgn(_filename):
    fens = []
    with open(_filename, 'r') as pgn:
        game = chess.pgn.read_game(pgn)
        while not game.is_end():
            fen = game.board().fen()
            fen = fen.split(' ')
            if fen[1] == 'b':
                fen = flip_board(fen)
            fens.append(fen)
            game = game.variation(0)
    return fens

def flip_board(fen):
    ''' switch colors of pieces'''
    new_fen = ''
    for char in fen:
        if char.isupper():
            new_fen += char.lower()
        elif char.islower():
            new_fen += char.upper()
        else:
            new_fen += char

    return new_fen


read_pgn('ct-Aronian, Levon-Carlsen, Magnus-2014.2.4.pgn')
