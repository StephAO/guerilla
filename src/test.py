# Unit tests

import sys
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path + '/../helpers/')

import numpy as np
import data_handler as dh
import stockfish_eval as sf


###############################################################################
# Input Tests
###############################################################################

def stockfish_test():
    """
    Tests stockfish scoring script and score mapping.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    seconds = 1

    # Fens in INCREASING score value
    fens = [None] * 8
    fens[0] = dh.flip_board('3qr1Qk/pbpp2pp/1p5N/6b1/2P1P3/P7/1PP2PPP/R4RK1 b - - 0 1')  # White loses in 2 moves
    fens[1] = dh.flip_board('3qr1Qk/pbpp2pp/1p5N/6b1/2P1P3/P7/1PP2PPP/R4RK1 b - - 0 1')  # White loses in 1 move
    fens[2] = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1'  # Starting position
    fens[
        3] = 'r1bqkbnr/ppp2ppp/2p5/2Q1p3/4P3/3P1N2/PPP2PPP/RNB1K2R` w - - 0 5'  # Midgame where white is definitely winning
    fens[4] = '2qrr1n1/3b1kp1/2pBpn1p/1p2PP2/p2P4/1BP5/P3Q1PP/4RRK1 w - - 0 1'  # mate in 10 (white wins)
    fens[5] = '4Rnk1/pr3ppp/1p3q2/5NQ1/2p5/8/P4PPP/6K1 w - - 0 1'  # mate in 3 (white wins)
    fens[6] = '3qr2k/pbpp2pp/1p5N/3Q2b1/2P1P3/P7/1PP2PPP/R4RK1 w - - 0 1'  # mate in 2 (white wins)
    fens[7] = '3q2rk/pbpp2pp/1p5N/6b1/2P1P3/P7/1PP2PPP/R4RK1 w - - 0 1'  # mate in 1 (white wins)

    # Test white play next
    prev_score = float('-inf')
    for i, fen in enumerate(fens):
        score = sf.get_stockfish_score(fen, seconds=seconds)
        if score < prev_score:
            print "Failure: Fen (%s) scored %d while fen (%s) scored %d. The former should have a lower score." \
                  % (fens[i - 1], prev_score, fens[i], score)
            return False
        prev_score = score

    # Test black play next
    prev_score = float('-inf')
    for i, fen in enumerate(fens):
        score = sf.get_stockfish_score(dh.flip_board(fen), seconds=seconds)
        if score < prev_score:
            print "Failure: Fen (%s) scored %d while fen (%s) scored %d. The former should have a lower score." \
                  % (dh.flip_board(fens[i - 1]), prev_score, dh.flip_board(fens[i]), score)
            return False
        prev_score = score

    return True


def diag_input_test():
    """
    Tests that boards are properly converted to diagonals.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    # Notes on diagonal structure:
    #   Diagonal order is upwards (/) then downwards (\)
    #   There are 10 diagonals in total. Shorter diagonals are 0 padded.
    #   Diagonals vary in size from 6 to 8

    # Randomly selected fen
    fens = ['r2q1rk1/p3n1b1/1pn1b2p/2P1p3/2B1P1pN/P1P2pP1/1B1N1P1P/R2QR1K1 b - - 3 18']

    # Expected result


    # TODO


def channel_input_test():
    """
    Tests that boards are properly converted to channels.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    # Notes:
    #   Channel order is [p,r,n,b,q,k] white and then black
    #   Within a board the order is Rank 8 -> Rank 1 and File a -> File h

    # Fens and Expected Results
    fens = []
    corr = []

    ### Randomly selected fen ###
    rand_fen = 'r2q1rk1/1pp2ppp/p1np1n2/1B2p3/Pb2P1b1/3P1N2/1PP1NPPP/R1BQ1RK1 w - - 0 9'

    # Expected result
    rand_corr = np.zeros([8, 8, 12])

    # Positions
    pos = {}

    # White Pawn, Rook, Knights, Bishops, Queens then King
    pos[0] = [(4, 0), (4, 4), (5, 3), (6, 1), (6, 2), (6, 5), (6, 6), (6, 7)]
    pos[1] = [(7, 0), (7, 5)]
    pos[2] = [(5, 5), (6, 4)]
    pos[3] = [(3, 1), (7, 4)]
    pos[4] = [(7, 3)]
    pos[5] = [(7, 6)]

    # Black Pawn, Rook, Knights, Bishops, Queens then King
    pos[6] = [(1, 1), (1, 2), (1, 5), (1, 6), (1, 7), (2, 0), (2, 3), (3, 4)]
    pos[7] = [(0, 0), (0, 5)]
    pos[8] = [(2, 2), (2, 5)]
    pos[9] = [(4, 1), (4, 6)]
    pos[10] = [(0, 3)]
    pos[11] = [(0, 6)]

    for channel, loc in pos.iteritems():
        for rank, file in loc:
            rand_corr[rank][file][channel] = 1

    fens.append(rand_fen)
    corr.append(rand_corr)

    ### Completely filled board with each piece ###
    for i, piece in enumerate('PRNBQKprnbqk'):
        fill_fen = '/'.join([piece * 8 for _ in range(8)]) + ' w KQkq - 0 1'
        fill_corr = np.dstack([np.zeros([8, 8, i]), np.ones([8, 8, 1]), np.zeros([8, 8, 12 - i])])
        fens.append(fill_fen)
        corr.append(fill_corr)

    ### Empty board ###
    empty_fen = '/'.join(['8'] * 8) + ' w KQkq - 0 1'
    empty_corr = np.zeros([8, 8, 12])
    fens.append(empty_fen)
    corr.append(empty_corr)

    success = True
    for i in range(len(fens)):
        actual = dh.fen_to_channels(fens[i])
        if not np.array_equal(corr[i], actual):
            print "Failed converting fen to channels:"
            print fens[i]
            print "Expected:"
            for j in range(12):
                print corr[i][:, :, j]
            print "Received:"
            for j in range(12):
                print actual[:, :, j]

                success = False

    return success


def main():
    print "-------- Input Tests --------"
    print "Testing Stockfish handling..."
    stockfish_test()
    # print "Testing board to channels..."
    # channel_input_test()


if __name__ == '__main__':
    main()
