# Unit tests
import os
import pickle
import random as rnd
import sys
import traceback

import chess
import numpy as np
import tensorflow as tf
from guppy import hpy
from pkg_resources import resource_filename

import guerilla.data_handler as dh
import guerilla.play.neural_net as nn
import guerilla.train.stockfish_eval as sf
from guerilla.hyper_parameters import *
from guerilla.players import Guerilla
from guerilla.play.search import Search
from guerilla.train.teacher import Teacher

# TODO: split it up between play and train


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
    max_attempts = 3

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
        score = sf.get_stockfish_score(fen, seconds=seconds, num_attempt=max_attempts)
        if score < prev_score:
            print "Failure: Fen (%s) scored %f while fen (%s) scored %f. The former should have a lower score." \
                  % (fens[i - 1], prev_score, fens[i], score)
            return False
        if sf.sigmoid_array(score) < sf.sigmoid_array(prev_score):
            print "Failure: Fen (%s) scored %f while fen (%s) scored %f. The former should have a lower score." \
                  % (fens[i - 1], sf.sigmoid_array(prev_score), fens[i], sf.sigmoid_array(score))
            return False
        prev_score = score

    # Test black play next
    prev_score = float('-inf')
    for i, fen in enumerate(fens):
        score = sf.get_stockfish_score(fen, seconds=seconds, num_attempt=max_attempts)
        if score < prev_score:
            print "Failure: Fen (%s) scored %f while fen (%s) scored %f. The former should have a lower score." \
                  % (dh.flip_board(fens[i - 1]), prev_score, dh.flip_board(fens[i]), score)
            return False
        if sf.sigmoid_array(score) < sf.sigmoid_array(prev_score):
            print "Failure: Fen (%s) scored %f while fen (%s) scored %f. The former should have a lower score." \
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
    rand_corr = np.zeros([10, 8, 12])
    for channel in range(rand_channels.shape[2]):
        idx = 0
        for i in range(-2, 3):
            unpad_up = np.diagonal(rand_channels[:, :, channel], offset=i)
            unpad_down = np.diagonal(np.flipud(rand_channels[:, :, channel]), offset=i)
            rand_corr[2 * idx, :, channel] = np.lib.pad(unpad_up, (0, 8 - len(unpad_down)), 'constant',
                                                        constant_values=0)
            rand_corr[2 * idx + 1, :, channel] = np.lib.pad(unpad_down, (0, 8 - len(unpad_down)),
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


def nsv_test(num_check=40, max_step=5000, tolerance=1e-2, allow_err=0.3, score_repeat=3):
    """
    Tests that fens.nsv and sf_values.nsv file are properly aligned. Also checks that the FENS are "white plays next".
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
    max_wrong = num_check * allow_err

    with open(resource_filename('guerilla', 'data/extracted_data/fens.nsv'), 'r') as fens_file, \
            open(resource_filename('guerilla', 'data/extracted_data/sf_values.nsv'), 'r') as sf_file:
        fens_count = 0
        while fens_count < num_check and len(wrong) <= max_wrong:
            for i in range(rnd.randint(0, max_step)):
                fens_file.readline()
                sf_file.readline()

            # Get median stockfish score
            fen = fens_file.readline().rstrip()
            median_arr = []
            for i in range(score_repeat):
                median_arr.append(sf.get_stockfish_score(fen, seconds=seconds))

            # Convert to probability of winning
            expected = sf.sigmoid_array(np.median(median_arr))
            actual = float(sf_file.readline().rstrip())

            if abs(expected - actual) > tolerance:
                wrong.append("For FEN '%s' expected score of %f, got file score of %f." % (fen, expected, actual))

            if dh.black_is_next(fen):
                print "White does not play next in this FEN: %s" % fen
                return False

            fens_count += 1

    if len(wrong) > max_wrong:
        for info in wrong:
            print info

        return False

    return True


###############################################################################
# SEARCH TESTS
###############################################################################

def white_search_test():
    """ Tests that search functions properly when playing white. """

    fen_str = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"

    board = chess.Board(fen=fen_str)
    # Can't run deeper due to restricted evaluation function.
    shallow = Search(eval_fn=basic_test_eval, max_depth=1, search_mode='recipromax')
    score, move, _ = shallow.run(board)
    if (score == 0.6) and (str(move) == "b3b4"):
        return True
    else:
        print "White Search Test failed. Expected [Score, Move]: [0.6, b3b4] got: [%.1f, %s]" % (score, move)
        return False


def black_search_test():
    """ Tests that search functions properly when playing black. """

    fen_str = dh.flip_board("8/p7/1p6/8/8/1P6/P7/8 b ---- - 0 1")

    board = chess.Board(fen=fen_str)
    # Can't run deeper due to restricted evaluation function.
    shallow = Search(eval_fn=basic_test_eval, max_depth=1, search_mode='recipromax')
    score, move, _ = shallow.run(board)
    if (score == 0.6) and (str(move) == "b3b4"):
        return True
    else:
        print "Black Search Test failed. Expected [Score, Move]: [0.6, b3b4] got: [%.1f, %s]" % (score, move)
        return False


def checkmate_search_test():
    """
    Tests that checkmates are working.
    """

    s = Search(eval_fn=(lambda x: 1), search_mode='recipromax')
    white_wins = chess.Board('R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1')
    black_wins = chess.Board('8/8/8/8/8/2k5/1p6/rK6 w - - 0 1')
    result, _, _ = s.run(white_wins)
    if result != 0:
        print "Checkmate Search Test failed, invalid result for white checkmate."
        return False
    result, _, _ = s.run(black_wins)
    if result != 0:
        print "Checkmate search test failed, invalid result for black checkmate."
        return False

    return True


def minimax_pruning_test():
    """ Runs a basic minimax and pruning test on the search class. """

    # Made up starting positions with white pawns in a2 & b3 and black pawns in a7 & b6 (no kings haha)
    # This allows for only 3 nodes at depth 1, 9 nodes at depth 2, and 21 nodes at depth 3 (max)

    fen_str = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"

    board = chess.Board(fen=fen_str)
    # Can't run deeper due to restricted evaluatoin function.
    shallow = Search(eval_fn=minimax_test_eval, max_depth=3, search_mode='recipromax')
    score, move, _ = shallow.run(board)
    if (score == 0.6) and (str(move) == "b3b4"):
        return True
    else:
        print "Mimimax Test failed: Expected [Score, Move]: [6, b3b4] got: [%.1f, %s]" % (score, move)
        return False


# Used in white search, black search and checkmate search test
def basic_test_eval(fen):
    board_state, player, _, _, _, _ = fen.split(' ')
    if player != 'w':
        raise RuntimeError("This shouldn't happen! Evaluation should always be called with white next.")

    board_state = dh.flip_board(fen).split(' ')[0]

    if board_state == "8/p7/1p6/8/8/PP6/8/8":  # a2a3
        return 0.5
    elif board_state == "8/p7/1p6/8/1P6/8/P7/8":  # b3b4
        return 0.4
    elif board_state == "8/p7/1p6/8/P7/1P6/8/8":  # a2a4
        return 0.7
    else:
        raise RuntimeError("This definitely should not happen! Invalid board.")


# Used in minimax + pruning test
def minimax_test_eval(fen):
    """
    Arbitrary evaluation function for minimax test. Should yield move of b3b4 and value of 6.
    """
    board_state, player, _, _, _, _ = fen.split(' ')
    if player != 'w':
        raise RuntimeError("This shouldn't happen! Evaluation should always be called with white next.")

    board_state = dh.flip_board(fen).split(' ')[0]

    if board_state == "8/p7/8/1p6/P7/1P6/8/8":
        score = 0.5
    elif board_state == "8/p7/8/1p6/1P6/P7/8/8":
        score = 0.4
    elif board_state == "8/8/pp6/8/P7/1P6/8/8":
        score = 0.1
    elif board_state == "8/8/pp6/8/1P6/P7/8/8":
        score = 0.6
    elif board_state == "8/8/1p6/p7/P7/1P6/8/8":
        score = 0.8
    elif board_state == "8/8/1p6/p7/1P6/P7/8/8":
        print "WARNING1: This node should not be reached when using alpha-beta pruning!"
        score = 0.0
    elif board_state == "8/p7/8/1p6/PP6/8/8/8":
        score = 0.7
    elif board_state == "8/8/pp6/1P6/8/8/P7/8":
        score = 0.1
    elif board_state == "8/8/pp6/8/PP6/8/8/8":
        score = 0.3
    elif board_state == "8/8/1p6/P7/8/8/P7/8":
        score = 0.8
    elif board_state == "8/8/1p6/pP6/8/8/P7/8":
        print "WARNING2: This node should not be reached when using alpha-beta pruning!"
        score = 0.0
    elif board_state == "8/8/1p6/p7/PP6/8/8/8":
        print "WARNING3: This node should not be reached when using alpha-beta pruning!"
        score = 0.0
    elif board_state == "8/p7/8/1P6/8/1P6/8/8":
        score = 0.99
    elif board_state == "8/p7/8/Pp6/8/1P6/8/8":
        score = 0.6
    elif board_state == "8/8/pp6/P7/8/1P6/8/8":
        score = 0.1
    else:
        raise RuntimeError("This definitely should not happen! Invalid board.")

    return 1 - score


###############################################################################
# NEURAL NET TEST
###############################################################################

def save_load_weights_test(verbose=False):
    """
    Tests that neural net can properly save and load weights.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    test_file = 'save_load_weights_test.p'

    # Generate randomly initialized neural net
    test_nn = nn.NeuralNet(verbose=verbose)
    test_nn.start_session()
    test_nn.init_graph()
    weights = test_nn.get_weight_values()

    # Save neural net weights to file
    test_nn.save_weight_values(_filename=test_file)

    # Close session
    test_nn.close_session()

    # Load neural net weights from file
    test_nn = nn.NeuralNet(load_file=test_file, verbose=verbose)
    test_nn.start_session()
    test_nn.init_graph()
    new_weights = test_nn.get_weight_values()
    test_nn.close_session()

    # Remove test file
    os.remove(resource_filename('guerilla', 'data/weights/' + test_file))

    # Compare saved and loaded weights
    result_msg = diff_dict_helper(weights, new_weights)
    if result_msg:
        print "Weight did not match."
        print result_msg
        return False

    return True

def diff_dict_helper(old_dict, new_dict):
    """
    Compares two dictionaries of numpy.arrays. Useful for comparing weights and training variables.
    Input
        old_dict [Dictionary of numpy.arrays]
            One of the dictionary you'd like to compare.
        new_dict [Dictionary of numpy.arrays]
            The other dictionary you'd like to compare.
    Output:
        Result [None or String]
            Returns None if identical, otherwise returns error message.
    """

    for weight in old_dict.iterkeys():
        if isinstance(new_dict[weight], list) and isinstance(old_dict[weight], list):
            success = all([np.array_equal(old_dict[weight][i], new_dict[weight][i])
                           for i in range(len(old_dict[weight]))])
            success = success and (len(old_dict[weight]) == len(new_dict[weight]))
        elif type(new_dict[weight]) == type(old_dict[weight]):
            success = np.array_equal(np.array(old_dict[weight]), np.array(new_dict[weight]))
        else:
            success = False

        if not success:
            return "Mismatching entries for '%s': Expected:\n %s \n Received:\n %s\n" % (weight,
                                                                                           str(old_dict[weight]),
                                                                                str(new_dict[weight]))

    if len(old_dict) != len(new_dict):
        return "Different number of entries for '%s': Expected Length:\n %s \n Received Length:\n %s\n" % (weight,                                                                                                  len(old_dict),                                                                                                  len(new_dict))

    return None

###############################################################################
# TRAINING TESTS
###############################################################################

def training_test(verbose=False):
    """ 
    Runs training in variety of fashions.
    Checks crashing, decrease in cost over epochs, consistent output, and memory usage.
    """
    success = True
    # Set hyper params for mini-test
    hp['NUM_FEAT'] = 10
    hp['NUM_EPOCHS'] = 5
    hp['BATCH_SIZE'] = 50
    hp['VALIDATION_SIZE'] = 50
    hp['TRAIN_CHECK_SIZE'] = 50
    hp['TD_LRN_RATE'] = 0.00001  # Learning rate
    hp['TD_DISCOUNT'] = 0.7  # Discount rate

    for t_m in nn.NeuralNet.training_modes:
        if t_m == 'adagrad':
            hp['LEARNING_RATE'] = 0.00001
        elif t_m == 'adadelta':
            continue  # TODO remove when adadelta is fully implemented
            hp['LEARNING_RATE'] = 0.00001
        elif t_m == 'bootstrap':
            hp['LEARNING_RATE'] = 0.00001

        error_msg = ""
        try:
            with Guerilla('Harambe', 'w', training_mode=t_m, verbose=False) as g:
                g.search.max_depth = 1
                t = Teacher(g, test=True, verbose=False)
                t.set_bootstrap_params(num_bootstrap=500)  # 488037
                t.set_td_params(num_end=3, num_full=3, randomize=False, end_length=3, full_length=3, batch_size=5)
                t.set_sp_params(num_selfplay=1, max_length=3)
                t.sts_on = False
                t.sts_interval = 100

                pre_heap_size = hpy().heap().size
                t.run(['train_bootstrap', 'train_td_endgames', 'train_td_full', 'train_selfplay', ], training_time=60)
                post_heap_size = hpy().heap().size

            loss = pickle.load(open(resource_filename('guerilla', 'data/loss/loss_test.p'), 'rb'))
            # Wrong number of losses
            if len(loss['train_loss']) != hp['NUM_EPOCHS'] + 1 or len(loss['loss']) != hp['NUM_EPOCHS'] + 1:
                error_msg += "Some bootstrap epochs are missing training or validation losses.\n" \
                             "Number of epochs: %d,  Number of training losses: %d, Number of validation losses: %d" % \
                             (hp['NUM_EPOCHS'], len(loss['train_loss']), len(loss['loss']))
                success = False
            # Training loss went up
            if loss['train_loss'][0] <= loss['train_loss'][-1]:
                error_msg += "Bootstrap training loss went up. Losses:\n%s" % (loss['train_loss'])
                success = False
            # Validation loss went up
            if loss['loss'][0] <= loss['loss'][-1]:
                error_msg += "Bootstrap validation loss went up. Losses:\n%s" % (loss['loss'])
                success = False
            # Memory usage increased significantly
            if float(abs(post_heap_size - pre_heap_size)) / float(pre_heap_size) > 0.01:
                success = False
                error_msg += "Memory increasing significantly when running training.\n" \
                             "Starting heap size: %d bytes, Ending heap size: %d bytes. Increase of %f %%" \
                             % (pre_heap_size, post_heap_size,
                                100. * float(abs(post_heap_size - pre_heap_size)) / float(pre_heap_size))
        # Training failed                    
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_msg += "The following error occured during the training:" \
                         "\n  type:\n    %s\n\n  Error msg:\n    %s\n\n  traceback:\n    %s\n" % \
                         (str(exc_type).split('.')[1][:-2], exc_value, \
                          '\n    '.join(''.join(traceback.format_tb(exc_traceback)).split('\n')))
            success = False

        if not success:
            print "Training with type %s fails:\n%s" % (t_m, error_msg)

    return success

def learn_sts_test(mode = 'queen', thresh=0.9):
    """
    Tests that Guerilla can learn the best moves in the Strategic Test Suite (STS).
    Fetches all the STS epds. Takes the top moves and gives them high probability of winning.
    Then randomly generates the same number of bad moves and gives them a low probability of winning.
    Trains the guerilla on this set of data. Runs STS on the guerilla trained on this set of data.
    Should yield a high STS score, the test is succesful if the STS score is sufficiently high.
    Note:
        Conversion to probability is winning done by the function P(x) = 0.6 + x^2*(0.003) where x in [0, 10].
    Input:
        mode [String]
            STS mode(s) to train and test on. See Guerilla.Teacher.eval_sts()  documentation for options.
        thresh [Float]
            Minimum STS score (as a percent of the maximum score) necessary for the test to be considered a success.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    # Get EPDS and Scores
    if type(mode) is not list:
        mode = [mode]

    # vars
    board = chess.Board()

    # Run tests
    epds = []
    for test in mode:
        epds += Teacher.get_epds_by_mode(test)

    # Convert scores to probability of winning and build data
    fens = []
    values = []
    for i, epd in enumerate(epds):

        board, move_scores = Teacher.parse_epd(epd)

        for move, score in move_scores.iteritems():

            # Add good moves
            board.push(move) # apply move
            fen = board.fen()
            fens.append(fen if dh.white_is_next(fen) else dh.flip_board(fen))
            values.append(1)#0.6 + (score**2)*0.003)
            board.pop() # undo move

        # # Add random bad move
        # move = rnd.choice(list(set(board.legal_moves) - set(move_scores.iterkeys())))
        # board.push(move)  # apply move
        # fen = board.fen()
        # fens.append(fen if dh.white_is_next(fen) else dh.flip_board(fen))
        # values.append(0.25)
        # board.pop() # undo move

    print len(fens)

    # Set hyper parameters
    hp['NUM_EPOCHS'] = 50
    hp['BATCH_SIZE'] = 50
    hp['VALIDATION_SIZE'] = 50
    hp['TRAIN_CHECK_SIZE'] = 5
    hp['LEARNING_RATE'] = 0.0001
    hp['LOSS_THRESHOLD'] = -100 # Make it so it never stops by convergence since VALIDATION_SIZE = 0

    # Add extra evaluation boards
    fens += fens[-hp['VALIDATION_SIZE']:]
    values += values[-hp['VALIDATION_SIZE']:]

    # set to multiple of batch size
    if (len(fens) % hp['BATCH_SIZE']) != 0:
        fens = fens[:(-1) * (len(fens) % hp['BATCH_SIZE'])]
    values = values[:len(fens)]

    # Train and Test Guerilla
    with Guerilla('Harambe', 'w', training_mode='adagrad') as g:
        # Train
        g.search.max_depth = 1
        t = Teacher(g)
        t.train_bootstrap(fens, values)

        # Run STS Test
        result = Teacher.eval_sts(g, mode=mode)

    if float(result[0][0])/result[1][0] <= thresh:
        print "STS Scores was too low, got a score of %d/%d" % (result[0][0], result[1][0])
        return False

    return True


def load_and_resume_test(verbose=False):
    """
    Tests the load_and_resume functionality of teacher.
    Things it checks for:
        (1) Doesn't crash.
        (2) Weights are properly loaded.
        (3) Graph training variables are properly loaded.
        (4) Correct action is loaded.
        (5) Correct set of actions is loaded.
        (6) Correct number of epochs.
    Does not check (among other things):
        (-) That all the necessary components of the training state are stored.
        (-) That the correct sequence of training actions is taken.
        (-) That training reduces the loss.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    # Modify hyperparameters for a small training example.
    success = True
    hp['NUM_FEAT'] = 10
    hp['NUM_EPOCHS'] = 5
    hp['BATCH_SIZE'] = 5
    hp['VALIDATION_SIZE'] = 5
    hp['TRAIN_CHECK_SIZE'] = 5
    hp['TD_LRN_RATE'] = 0.00001  # Learning rate
    hp['TD_DISCOUNT'] = 0.7  # Discount rate
    hp['LEARNING_RATE'] = 0.00001

    # Pickle path
    loss_path = resource_filename('guerilla', 'data/loss/')

    # Test for each training type & all training types together
    train_actions = Teacher.actions[:-1]
    train_actions.append(Teacher.actions[:-1])
    for action in train_actions:
        set_of_actions = action if isinstance(action, list) else [action]

        # Error message:
        error_msg = ''

        # Reset graph
        tf.reset_default_graph()

        # Run action
        with Guerilla('Harambe', 'w', verbose=verbose) as g:
            g.search.max_depth = 1
            t = Teacher(g, test=True, verbose=verbose)
            t.set_bootstrap_params(num_bootstrap=50)  # 488037
            t.set_td_params(num_end=3, num_full=3, randomize=False, end_length=2, full_length=2)
            t.set_sp_params(num_selfplay=3, max_length=5)

            # Run
            t.run(set_of_actions, training_time= (0.5 if not isinstance(action, list) else 4))

            # Save current action
            pause_action = t.actions[t.curr_action_idx]

            # Save Weights
            weights = g.nn.get_weight_values()

            # Save graph training variables
            train_vars = g.nn.sess.run(g.nn.get_training_vars())

        # Reset graph
        tf.reset_default_graph()

        # Run resume
        with Guerilla('Harambe', 'w', verbose=verbose) as g:
            g.search.max_depth = 1
            t = Teacher(g, test=True, verbose=verbose)
            t.set_bootstrap_params(num_bootstrap=500)  # 488037

            # Run
            t.run(['load_and_resume'])

            # Save loaded current action
            state = t.load_state() # resets weights and training vars to start of resume values
            new_actions = state['actions']
            new_action = new_actions[state['curr_action_idx']]

            # Get new weights
            new_weights = g.nn.get_weight_values()

            # Save new training variables
            new_train_vars = g.nn.sess.run(g.nn.get_training_vars())


        # Compare weight values
        result_msg = diff_dict_helper(weights, new_weights)
        if result_msg:
            error_msg += "Weight did not match.\n"
            error_msg += result_msg
            success = False

        # Compare graph training variable values
        result_msg = diff_dict_helper(train_vars, new_train_vars)
        if result_msg:
            error_msg += "Training variables did not match.\n"
            error_msg += result_msg
            success = False

        # Compare the action
        if pause_action != new_action:
            error_msg += "Current action was not saved and loaded properly. \n"
            error_msg += "Saved:\n %s \n Loaded:\n %s\n" % (pause_action, new_action)
            success = False

        # Compare the set of actions
        if set_of_actions != new_actions:
            error_msg += "Set of actions was not saved and loaded properly. \n"
            error_msg += "Saved:\n %s \n Loaded:\n %s\n" % (str(set_of_actions), str(new_actions))
            success = False

        # Check that correct number of epochs is run
        with open(loss_path + 'loss_test.p', 'r') as f:
            loss = pickle.load(f)
            if hp['NUM_EPOCHS'] != (len(loss['loss']) - 1):
                error_msg += "On action %s there was the wrong number of epochs. " % action
                error_msg += "Expected %d epochs, but got %d epochs." % (hp['NUM_EPOCHS'], len(loss['loss']) - 1)
                success = False

        if not success:
            print "Load and resume with action %s fails:\n%s" % (str(action), error_msg)

    return success

def main():
    all_tests = {}
    all_tests["Input Tests"] = {'Stockfish Handling': stockfish_test,
                                'Board to Channels': channel_input_test,
                                'Channels to Diagonals': diag_input_test,
                                'NSV Alignment': nsv_test}

    all_tests["Search Tests"] = {'White Search': white_search_test,
                                 'Black Search': black_search_test,
                                 'Checkmate Search': checkmate_search_test,
                                 'Minimax And Pruning': minimax_pruning_test}

    all_tests["Neural Net Tests"] = {'Weight Save and Load': save_load_weights_test}

    all_tests["Training Tests"] = {'Training': training_test,
                                   'Load and Resume': load_and_resume_test,
                                   'Learn STS':learn_sts_test}

    success = True
    for group_name, group_dict in all_tests.iteritems():
        print "-------- " + group_name + " --------"
        for name, test in group_dict.iteritems():
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
