# TODO: double check that there aren't unecessary imports
import sys
import os
import time
import chess
import numpy as np
from guerilla.hyper_parameters import *


def flip_board(fen):
    """ switch colors of pieces
        input:
            fen:
                fen string (only board state)
        output:
            new_fen:
                fen string with colors switched
    """
    board_fen, turn, castling, en_passant, half_clock, full_clock = fen.split()

    # board fen
    new_board_fen = ''
    for char in board_fen:
        if char.isupper():
            new_board_fen += char.lower()
        elif char.islower():
            new_board_fen += char.upper()
        else:
            new_board_fen += char
    new_fen_list = new_board_fen.split('/')
    new_board_fen = '/'.join(new_fen_list[::-1])

    # turn
    turn = 'w' if turn == 'b' else 'b'

    # castling
    new_white_castling = ''
    new_black_castling = ''
    for char in castling:
        if char.islower():
            new_white_castling += char.upper()
        else:
            new_black_castling += char.lower()
    new_castling = new_white_castling + new_black_castling

    # en_passant
    new_en_passant = ''
    if en_passant != '-':
        new_en_passant += en_passant[0]
        new_en_passant += str(9 - int(en_passant[1]))
    else:
        new_en_passant = '-'

    return ' '.join((new_board_fen, turn, new_castling, new_en_passant, half_clock, full_clock))

# TODO: deal with en passant and castling
def fen_to_channels(fen):
    """
        Converts a fen string to channels for neural net.
        Always assumes that it's white's turn

        Inputs:
            fen[string]:
                fen string describing current state. 
                Currently only using board state

        Output:
            Channels[ndarray]:

                Consists of 12 8x8 channels (12 8x8 chess boards)
                12 Channels: 6 for each you and your opponents piece types
                Types in order are: Pawns, Rooks, Knights, Bishops, Queens, King
                First 6 channels are white pieces, last 6 are black.
                Index [0,0] corresponds to rank 1 and file a; [8,8] to rank 8 and file h.
    """

    # fen = fen.split(' ')
    # board_str = fen[0]
    # turn = fen[1]
    # castling = fen[2]
    # en_passant = fen[3]

    channels = np.zeros((8, 8, hp['NUM_CHANNELS']))

    c_file = 0
    c_rank = 7
    for char in fen:
        if char == ' ':
            break
        if char == '/':
            c_file = 0
            c_rank -= 1
            continue
        elif char.isdigit():
            c_file += int(char)
            continue
        else:
            white = char.isupper()
            char = char.lower()
            if white:
                channels[c_rank, c_file, piece_indices[char]] = 1
            else:
                channels[c_rank, c_file, piece_indices[char] + 6] = 1

        c_file += 1
        if c_rank == 0 and c_file == 8:
            break
    return channels

def get_diagonals(channels):
    """
        Retrieves and returns the diagonals from the board

        Ouput:
            Diagonals[ndarray]:
                12 Channels: 6 for each you and your opponents piece types
                Types in order are: Pawns, Rooks, Knights, Bishops, Queens, King
                First 6 channels are your pieces, last 6 are opponents.
                Each piece array has 10 diagonals with max size of 8 (shorter diagonals are 0 padded at the end)
                Diagonal ordering is a3 up, a6 down, a2 up, a7 down, a1 up, a8 down, b1 up, b8 down, c1 up, c8 down
    """
    diagonals = np.zeros((10, 8, hp['NUM_CHANNELS']))
    for i in xrange(hp['NUM_CHANNELS']):
        index = 0
        for o in xrange(-2,3):
            diag_up = np.diagonal(channels[:, :, i], offset=o)
            diag_down = np.diagonal(np.flipud(channels[:, :, i]), offset=o)

            diagonals[index, 0 : 8 - abs(o), i] = diag_up
            index += 1
            diagonals[index, 0 : 8 - abs(o), i] = diag_down
            index += 1

    return diagonals


def sigmoid_array(values):
    """ From: http://chesscomputer.tumblr.com/post/98632536555/using-the-stockfish-position-evaluation-score-to
        1000 cp lead almost guarantees a win (a sigmoid within that). From the looking at the graph to gather a few
        data points and using a sigmoid curve fitter an inaccurate function of 1/(1+e^(-0.00547x)) was decided on
        (by me, deal with it). Ideally this fitter function is learned, but this is just for testing so..."""
    return 1. / (1. + np.exp(-0.00547 * values))


def white_is_next(fen):
    """
    Returns true if fen is for white playing next.
    Inputs:
        fen [String]
            Chess board FEN.
    Output:
        [Boolean]
            True if fen is for white playing next.
    """
    if fen.split(' ')[1] == 'w':
        return True
    return False


def black_is_next(fen):
    """
    Returns true if fen is for black playing next.
    Inputs:
        fen [String]
            Chess board FEN.
    Output:
        [Boolean]
            True if fen is for black playing next.
    """
    return not white_is_next(fen)

def main():
    test_channel = fen_to_channels(chess.STARTING_FEN)
    start_time = time.clock()
    for i in xrange(10000):
        get_diagonals(test_channel)
    print '10000 iterations of get_diagonals:', time.clock() - start_time

if __name__ == '__main__':
    main()