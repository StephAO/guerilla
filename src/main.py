import sys
import os
dir_path = os.path.dirname(__file__)
sys.path.insert(0, dir_path + '/../helpers/')

import chess
import numpy as np
import stockfish_eval as sf
import chess_game_parser as cgp
import neural_net as nn
from hyper_parameters import *


def fen_to_channels(fen):
    """
        Converts a fen string to channels for neural net.
        Always assumes that it's white's turn
        @TODO deal with en passant and castling

        Inputs:
            fen[string]:
                fen string describing current state. 
                Currently only using board state

        Output:
            Channels[ndarray]:

                Consists of 12 8x8 channels (12 8x8 chess boards)
                12 Channels: 6 for each you and your opponents piece types
                Types in order are: Pawns, Rooks, Knights, Bishops, Queens, King
                First 6 channels are your pieces, last 6 are opponents.
    """

    # fen = fen.split(' ')
    # board_str = fen[0]
    # turn = fen[1]
    # castling = fen[2]
    # en_passant = fen[3]

    channels = np.zeros((8,8,NUM_CHANNELS))

    file = 0
    rank = 0
    empty_char = False
    for char in fen:
        if char == '/':
            file = 0
            rank += 1
            continue
        elif char.isdigit():
            file += int(char)
            continue
        else:
            my_piece = char.islower() # double check this. Normal fen, black is lower, but stockfish seems use to lower as current move
            char = char.lower()
            if my_piece:
                channels[rank, file, piece_indices[char]] = 1
            else:
                channels[rank, file, piece_indices[char] + 6] = 1

            # channels[rank, file, piece_indices[char] + 12] = 1 if my_piece else -1
        file += 1
        if rank == 7 and file == 8:
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
                Each piece array has 10 diagonals with max size of 8 
                (shorter diagonasl are 0 padded)
    """
    diagonals = np.zeros((10, 8, NUM_CHANNELS))
    for piece_idx in piece_indices.values():

        # diagonals with length 6 and 7
        for length in xrange(6,8):
            for i in xrange(length):
                offset = 8-length
                diag_offset = 4 if length == 7 else 0
                for channel in xrange(NUM_CHANNELS):
                    # upwards diagonals
                    diagonals[0+diag_offset, int(offset/2)+i, channel] = channels[i+offset, i, channel]
                    diagonals[1+diag_offset, int(offset/2)+i, channel] = channels[i, i+offset, channel]
                    #downwards diagonals
                    diagonals[2+diag_offset, int(offset/2)+i, channel] = channels[7-offset-i, i, channel]
                    diagonals[3+diag_offset, int(offset/2)+i, channel] = channels[7-i, offset-i, channel]

        # diagonals with length 8
        for i in xrange(8):
            for channel in xrange(NUM_CHANNELS):
                # upwards
                diagonals[8, i, channel] = channels[i, i, channel]
                # downwards
                diagonals[9, i, channel] = channels[7-i, i, channel]

    return diagonals

def get_stockfish_values(boards):
    ''' 
        Uses stockfishes evaluation to get a score for each board, then uses a sigmoid to map
        the scores to a winning probability between 0 and 1 (see sigmoid_array for how the sigmoid was chosen)

        Inputs:
            boards[list of strings]:
                list of board fens

        Outputs:
            values[list of floats]:
                a list of values for each board ranging between 0 and 1
    '''        
    cps = []
    i = 0
    for b in boards:
    # cp = centipawns advantage
        cp = sf.stockfish_scores(b, seconds=2)
        print cp
        if cp is not None:
            cps.append(cp)
    cps = np.array(cps)
    print np.shape(cps)
    return sigmoid_array(cps)



def sigmoid_array(values):
    ''' From: http://chesscomputer.tumblr.com/post/98632536555/using-the-stockfish-position-evaluation-score-to
        1000 cp lead almost guarantees a win (a sigmoid within that). From the looking at the graph to gather a few data point
        and using a sigmoid curve fitter an inaccurate function of 1/(1+e^(-0.00547x)) was decided on (by me, deal with it)
        Ideally this fitter function is learned, but this is just for testing so...'''
    return 1./(1. + np.exp(-0.00547*values))


train = True

fens = cgp.load_fens()

num_batches = len(fens)/BATCH_SIZE

true_values = sf.load_stockfish_values()
true_values = np.reshape(true_values[:num_batches*BATCH_SIZE], (num_batches, BATCH_SIZE))
    
print "Finished getting stockfish values. Begin training neural_net with %d items" % (len(fens))

boards = np.zeros((num_batches, BATCH_SIZE, 8, 8, NUM_CHANNELS))
diagonals = np.zeros((num_batches, BATCH_SIZE, 10, 8, NUM_CHANNELS))

net = nn.NeuralNet(load_weights=(not train))

if train:
    raw_input('This will overwrite your old weights\' pickle, do you still want to proceed? (Hit Enter)')
    print 'Training data. Will save weights to pickle'

    for i in xrange(num_batches*BATCH_SIZE):
        batch_num = i/BATCH_SIZE
        batch_idx = i % BATCH_SIZE
        boards[batch_num][batch_idx] = fen_to_channels(fens[i])

        for i in xrange(BATCH_SIZE):
            diagonals[batch_num][batch_idx] = get_diagonals(boards[batch_num][batch_idx])
    nn.train(net, boards, diagonals, true_values)

nn.evaluate(net, boards, diagonals, true_values)

