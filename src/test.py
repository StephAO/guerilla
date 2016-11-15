# Unit tests

import sys
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path + '/../helpers/')

import numpy as np
import random as rnd
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
    fens[3] = 'r1bqkbnr/ppp2ppp/2p5/2Q1p3/4P3/3P1N2/PPP2PPP/RNB1K2R w - - 0 5'  # Midgame where white is winning
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
        if sf.sigmoid_array(score) < sf.sigmoid_array(prev_score):
            print "Failure: Fen (%s) scored %d while fen (%s) scored %d. The former should have a lower score." \
                  % (fens[i - 1], sf.sigmoid_array(prev_score), fens[i], sf.sigmoid_array(score))
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
        if sf.sigmoid_array(score) < sf.sigmoid_array(prev_score):
            print "Failure: Fen (%s) scored %d while fen (%s) scored %d. The former should have a lower score." \
                  % (dh.flip_board(fens[i - 1]), sf.sigmoid_array(prev_score), dh.flip_board(fens[i]),
                     sf.sigmoid_array(score))
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
    #   Diagonal order is upwards (/) then downwards (\) with respect to chess board (origin is bottom left)
    #   There are 10 diagonals in total. Shorter diagonals are 0 padded at the end.
    #   Diagonals vary in size from 6 to 8
    #   Diagonal ordering is a3 up, a6 down, a2 up, a7 down, a1 up, a8 down, b1 up, b8 down, c1 up, c8 down

    # Fens and Expected Results
    fens = []
    corr = []

    ### Randomly selected fen ###
    rand_fen = 'r2q1rk1/1pp2ppp/p1np1n2/1B2p3/Pb2P1b1/3P1N2/1PP1NPPP/R1BQ1RK1 w - - 0 9'

    # Expected result
    rand_channels = np.zeros([8, 8, 12])

    # Positions
    pos = {}

    # White Pawn, Rook, Knights, Bishops, Queens then King
    pos[0] = [(3, 0), (3, 4), (2, 3), (1, 1), (1, 2), (1, 5), (1, 6), (1, 7)]
    pos[1] = [(0, 0), (0, 5)]
    pos[2] = [(2, 5), (1, 4)]
    pos[3] = [(4, 1), (0, 2)]
    pos[4] = [(0, 3)]
    pos[5] = [(0, 6)]

    # Black Pawn, Rook, Knights, Bishops, Queens then King
    pos[6] = [(6, 1), (6, 2), (6, 5), (6, 6), (6, 7), (5, 0), (5, 3), (4, 4)]
    pos[7] = [(7, 0), (7, 5)]
    pos[8] = [(5, 2), (5, 5)]
    pos[9] = [(3, 1), (3, 6)]
    pos[10] = [(7, 3)]
    pos[11] = [(7, 6)]

    for channel, loc in pos.iteritems():
        for rank, file in loc:
            rand_channels[rank][file][channel] = 1

    # Convert to diagonals
    rand_corr = np.zeros([10,8,12])
    for channel in range(rand_channels.shape[2]):
        idx = 0
        for i in range(-2,3):
            unpad_up = np.diagonal(rand_channels[:,:,channel], offset=i)
            unpad_down = np.diagonal(np.flipud(rand_channels[:, :, channel]), offset=i)
            rand_corr[2*idx,:,channel] = np.lib.pad(unpad_up, (0, 8 - len(unpad_down)), 'constant', constant_values=0)
            rand_corr[2*idx + 1,:,channel] = np.lib.pad(unpad_down, (0, 8 - len(unpad_down)),
                                                        'constant', constant_values=0)
            idx += 1

    fens.append(rand_fen)
    corr.append(rand_corr)

    ### Completely filled board with each piece ###
    diag_vecs = [[1] * 6 + [0] * 2, [1] * 7 + [0], [1] * 8, [1] * 7 + [0], [1] * 6 + [0] * 2]
    filled_diag = np.repeat(np.stack(diag_vecs, axis=0), 2, axis=0)
    for i, piece in enumerate('PRNBQKprnbqk'):
        fill_fen = '/'.join([piece * 8 for _ in range(8)]) + ' w KQkq - 0 1'
        fill_corr = np.dstack([np.zeros([10, 8, i]), filled_diag, np.zeros([10, 8, 11 - i])])
        fens.append(fill_fen)
        corr.append(fill_corr)

    ### Empty board ###
    empty_fen = '/'.join(['8'] * 8) + ' w KQkq - 0 1'
    empty_corr = np.zeros([10, 8, 12])
    fens.append(empty_fen)
    corr.append(empty_corr)

    success = True
    for i in range(len(fens)):
        actual = dh.get_diagonals(dh.fen_to_channels(fens[i]))
        if not np.array_equal(corr[i], actual):
            print "Failed converting fen to diagonals:"
            print fens[i]
            print "Expected:"
            for j in range(12):
                print corr[i][:, :, j]
            print "Received:"
            for j in range(12):
                print actual[:, :, j]

                success = False

    return success


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
    pos[0] = [(3, 0), (3, 4), (2, 3), (1, 1), (1, 2), (1, 5), (1, 6), (1, 7)]
    pos[1] = [(0, 0), (0, 5)]
    pos[2] = [(2, 5), (1, 4)]
    pos[3] = [(4, 1), (0, 2)]
    pos[4] = [(0, 3)]
    pos[5] = [(0, 6)]

    # Black Pawn, Rook, Knights, Bishops, Queens then King
    pos[6] = [(6, 1), (6, 2), (6, 5), (6, 6), (6, 7), (5, 0), (5, 3), (4, 4)]
    pos[7] = [(7, 0), (7, 5)]
    pos[8] = [(5, 2), (5, 5)]
    pos[9] = [(3, 1), (3, 6)]
    pos[10] = [(7, 3)]
    pos[11] = [(7, 6)]

    for channel, loc in pos.iteritems():
        for rank, file in loc:
            rand_corr[rank][file][channel] = 1

    fens.append(rand_fen)
    corr.append(rand_corr)

    ### Completely filled board with each piece ###
    for i, piece in enumerate('PRNBQKprnbqk'):
        fill_fen = '/'.join([piece * 8 for _ in range(8)]) + ' w KQkq - 0 1'
        fill_corr = np.dstack([np.zeros([8, 8, i]), np.ones([8, 8, 1]), np.zeros([8, 8, 11 - i])])
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

def nsv_test(num_check=10, max_step=15, tolerance=1e-2, allow_err=0.3, score_repeat=3):
    """
    Tests that fens.nsv and sf_values.nsv file are properly aligned.
    NOTE: Need at least num_check*max_step stockfish and fens stored in the nsv's.
    Input:
        num_check [Int]
            Number of fens to check.
        max_step [Int]
            Maximum size of the random line jump within a file. i.e. at most, the next line checked will be max_step
            lines away from the current line.
        tolerance [Float]
            How far away the expected stockfish score can be from the actual stockfish score. 0 < tolerance < 1
        allow_err [Float]
            The percentage of mismatching stockfish scores allowed. 0 < allow_err < 1
        score_repeat [Int]
            Each stockfish scoring is repeated score_repeat times and the median is taken. Allows for variations in
            available memory.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    # Number of seconds spent on each stockfish score
    seconds = 1
    wrong = []
    max_wrong = num_check*allow_err

    with open(dir_path +'/../helpers/extracted_data/fens.nsv') as fens_file, \
            open(dir_path +'/../helpers/extracted_data/sf_values.nsv') as sf_file:
        fens_count = 0
        while fens_count < num_check and len(wrong) <= max_wrong:
            for i in range(rnd.randint(0,max_step)):
                fens_file.readline()
                sf_file.readline()

            # Get median stockfish score
            fen = fens_file.readline().rstrip()
            median_arr=[]
            for i in range(score_repeat):
                median_arr.append(sf.get_stockfish_score(fen, seconds=seconds))

            # Convert to probability of winning
            expected = sf.sigmoid_array(np.median(median_arr))
            actual = float(sf_file.readline().rstrip())

            if abs(expected - actual) > tolerance:
                wrong.append("For FEN '%s' expected score of %f, got file score of %f." % (fen, expected, actual))

            fens_count += 1

    if len(wrong) > max_wrong:
        for info in wrong:
            print info

        return False

    return True

def main():
    print "-------- Input Tests --------"
    input_tests = {'Stockfish handling': stockfish_test,
                   'Board to channels': channel_input_test,
                   'Channels to diagonals': diag_input_test,
                   'NSV alignment': nsv_test
                   }
    success = True
    for name, test in input_tests.iteritems():
        print "Testing " + name + "..."
        if not test():
            print "%s test failed" % name.capitalize()
            success = False

    if success:
        print "All tests passed"
    else:
        print "You broke something - go fix it"


if __name__ == '__main__':
    main()
