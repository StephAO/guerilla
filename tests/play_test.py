# Play Unit tests
import os
import random
import sys
import time
import Queue

import chess
import numpy as np
from pkg_resources import resource_filename

import guerilla.data_handler as dh
import guerilla.play.neural_net as nn
from guerilla.play.search import *
from guerilla.play.search_helpers import *


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
    test_nn.reset_graph()

    # Load neural net weights from file
    test_nn = nn.NeuralNet(load_file=test_file, verbose=verbose)
    test_nn.start_session()
    test_nn.init_graph()
    new_weights = test_nn.get_weight_values()
    test_nn.close_session()
    test_nn.reset_graph()


    # Remove test file
    os.remove(resource_filename('guerilla', 'data/weights/' + test_file))

    # Compare saved and loaded weights
    result_msg = dh.diff_dict_helper([weights, new_weights])
    if result_msg:
        print "Weight did not match."
        print result_msg
        return False

    return True

###############################################################################
# SEARCH TESTS
###############################################################################

def k_bot_test():
    """
    Tests the k-bot function used in Rank-Prune searching.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    test_list = [10, 79, 9, 59, 9, 47, 50, 41, 36, 80, 63, 25, 76, 81, 81, 30, 79, 81, 26, 52]
    test_list_sorted = sorted(test_list)

    success = True
    for test in [1, 3, 5, 10]:
        result = k_bot(list(test_list), test)
        if len(result) != test:
            print "Error: k_bot does not return the correct number of items.\n" \
                  "Expected: %d, Actual: %d" % (test, len(result))
            success = False
        if set(result) != set(test_list_sorted[:test]):
            print "Error: k_bot does not return the %d smallest values\n" \
                  "Expected: %s, Got: %s" % (test, str(test_list_sorted[:test]), str(result))
            success = False

    return success


def partition_test(num_test=5, seed=12345):
    """
    Tests the partition function used in Rank-Prune searching.
    Input:
        num_test [Int] (Optional)
            The number of random tests performed.
        seed [Int] (Optional)
            Seed for random number generator. For reproducibility.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    np.random.seed(seed)
    success = True

    for _ in range(num_test):
        arr_len = np.random.randint(0, 20)
        rnd_arr = np.random.randint(0, 100, arr_len)
        rnd_pivot = np.random.randint(0, arr_len)
        pivot_val = rnd_arr[rnd_pivot]

        pivot_idx = partition(rnd_arr, 0, arr_len - 1, rnd_pivot)

        # Check that pivot was moved to correct place
        if rnd_arr[pivot_idx] != pivot_val:
            print "Partition Test Failed: Expected pivot value of %d, got %d." % (pivot_val, rnd_arr[pivot_idx])
            success = False

        # Check that partitioning was successful
        for i in range(arr_len):
            if i < pivot_idx and rnd_arr[i] >= pivot_val:
                print "Partition Test Failed: (rnd_arr[%d] = %d) >= (pivot_idx = %d, pivot_val = %d)" % (i, rnd_arr[i],
                                                                                                         pivot_idx,
                                                                                                         pivot_val)
                success = False
            elif i > pivot_idx and rnd_arr[i] < pivot_val:
                print "Partition Test Failed: (rnd_arr[%d] = %d) < (pivot_idx = %d, pivot_val = %d)" % (i, rnd_arr[i],
                                                                                                        pivot_idx,
                                                                                                        pivot_val)
                success = False

        return success


def quickselect_test(num_test=10, seed=12345):
    """
    Tests the quickselect function used in Rank-Prune searching.
    Input:
        num_test [Int] (Optional)
            The number of random tests performed.
        seed [Int] (Optional)
            Seed for random number generator. For reproducibility.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    np.random.seed(seed)

    for _ in range(num_test):
        arr_len = np.random.randint(1, 20)
        rnd_arr = np.random.randint(0, 100, arr_len)
        k = np.random.randint(0, arr_len)

        quickselect(rnd_arr, 0, arr_len - 1, k)

        expected = sorted(rnd_arr)[k]
        if rnd_arr[k] != expected:
            print "Quickselect Test Failed: Expected %d-th smallest element to be %d, got %d" % (
                k, expected, rnd_arr[k])
            return False

    return True


def iterative_prune_test():
    success = True
    fen = '7k/8/5P2/8/8/8/P7/3K4 w - - 0 1'
    ip = IterativeDeepening(iterative_prune_test_eval, h_prune=True, prune_perc=0.5,
                            max_depth=2, time_limit=1e6)
    score, best_move, leaf_board = ip.run(chess.Board(fen))
    if score != 0.8:
        success = False
        print "Error: Wrong score, Expected: 0.8, Actual %s" % (score)
    if str(best_move) != "f6f7":
        success = False
        print "Error: Wrong best move, Expected: f6f7, Actual: %s" % (str(best_move))
    if leaf_board.split()[0] != "8/5P1k/8/8/8/8/P7/3K4":
        success = False
        print "Error: Wrong leaf board, Expected: 8/5P1k/8/8/8/8/P7/3K4, Actual %s" % (leaf_board.split()[0])
    curr_layer = [(None, ip.root)]
    next_layer = []
    for i in xrange(3):
        for child in curr_layer:
            move = child[0]
            node = child[1]
            fen = dh.flip_to_white(node.fen)
            if fen.split()[0] =='7k/8/5P2/8/8/8/P7/3K4':
                if abs(node.value - 0.8) > 0.0001 or node.depth != 0 or not node.expand:
                    success = False
                    print "Error: Node with fen 7k/8/5P2/8/8/8/P7/3K4 incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif fen.split()[0] =='2k5/p7/8/8/8/5p2/8/7K':
                if abs(node.value - 0.6) > 0.0001 or node.depth != 1 or node.expand:
                    success = False
                    print "Error: Node with fen 2k5/p7/8/8/8/5p2/8/7K incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif fen.split()[0] =='4k3/p7/8/8/8/5p2/8/7K':
                if abs(node.value - 0.6) > 0.0001 or node.depth != 1 or node.expand:
                    success = False
                    print "Error: Node with fen 4k3/p7/8/8/8/5p2/8/7K incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif fen.split()[0] =='8/p1k5/8/8/8/5p2/8/7K':
                if abs(node.value - 0.6) > 0.0001 or node.depth != 1 or node.expand:
                    success = False
                    print "Error: Node with fen 8/p2k4/8/8/8/5p2/8/7K incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif fen.split()[0] =='8/p2k4/8/8/8/5p2/8/7K':
                if abs(node.value - 0.6) > 0.0001 or node.depth != 1 or node.expand:
                    success = False
                    print "Error: Node with fen 8/p1k5/8/8/8/5p2/8/7K incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif fen.split()[0] =='8/p3k3/8/8/8/5p2/8/7K':
                if abs(node.value - (-0.7)) > 0.0001 or node.depth != 1 or not node.expand:
                    success = False
                    print "Error: Node with fen 8/p3k3/8/8/8/5p2/8/7K is incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif fen.split()[0] =='3k4/8/p7/8/8/5p2/8/7K':
                if abs(node.value - (-0.7)) > 0.0001 or node.depth != 1 or not node.expand:
                    success = False
                    print "Error: Node with fen 3k4/8/p7/8/8/5p2/8/7K is incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif fen.split()[0] =='3k4/p7/8/8/8/8/5p2/7K':
                if abs(node.value - (-0.8)) > 0.0001 or node.depth != 1 or not node.expand:
                    success = False
                    print "Error: Node with fen 3k4/p7/8/8/8/8/5p2/7K is incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif fen.split()[0] =='3k4/8/8/p7/8/5p2/8/7K':
                if abs(node.value - (-0.7)) > 0.0001 or node.depth != 1 or not node.expand:
                    # Note: that h8g8 is checked before h8h7 due to KILLER move ordering
                    success = False
                    print "Error: Node with fen 3k4/8/8/p7/8/5p2/8/7K is incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif str(move) =='h8g8':
                if abs(node.value - 0.7) > 0.0001 or node.depth != 2:
                    success = False
                    print "Error: Node with fen 8/7k/5P2/8/8/8/P7/2K5 is incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif str(move) =='h8h7':
                if abs(node.value - 0.8) > 0.0001 or node.depth != 2:
                    print node.fen, move
                    success = False
                    print "Error: Node with fen 6k1/8/5P2/8/8/8/P7/2K5 is incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)

            elif str(move) == 'h8g7':
                if abs(node.value - 0.9) > 0.0001 or node.depth != 2:
                    success = False
                    print "Error: Node with fen 8/5Pk1/8/8/8/8/P7/3K4 is incorrect (value: %f, depth: %d, expand: %s)" % (
                    node.value, node.depth, node.expand)
            else:
                success = False
                print "Error: Node with fen %s shouldn't exists, move: %s" % (fen, str(move))

            if node.expand:
                next_layer.extend(node.children.items())
        curr_layer = list(next_layer)
        next_layer = []

    return success

def iterative_prune_test_eval(fen):
    if fen.split()[1] == 'b':
        fen = dh.flip_board(fen)

    if dh.strip_fen(fen) == '7k/8/5P2/8/8/8/P7/3K4':
        return 0.0

    elif dh.strip_fen(fen) == '2k5/p7/8/8/8/5p2/8/7K':
        return 0.6

    elif dh.strip_fen(fen) == '4k3/p7/8/8/8/5p2/8/7K':
        return 0.6

    elif dh.strip_fen(fen) == '8/p1k5/8/8/8/5p2/8/7K':
        return 0.6

    elif dh.strip_fen(fen) == '8/p2k4/8/8/8/5p2/8/7K':
        return 0.6

    elif dh.strip_fen(fen) == '8/p3k3/8/8/8/5p2/8/7K':
        return 0.4

    elif dh.strip_fen(fen) == '3k4/8/p7/8/8/5p2/8/7K':
        return 0.4

    elif dh.strip_fen(fen) == '3k4/p7/8/8/8/8/5p2/7K':
        return 0.4

    elif dh.strip_fen(fen) == '3k4/8/8/p7/8/5p2/8/7K':
        return 0.4

    elif dh.strip_fen(fen)[:8] == '8/7k/5P2':
        return 0.8

    elif dh.strip_fen(fen)[:9] == '6k1/8/5P2':
        return 0.7

    elif dh.strip_fen(fen)[:8] == '8/5Pk1/8':
        return 0.9

    elif dh.strip_fen(fen)[:8] == '8/5P1k/8':
        return 0.8

    else:
        raise RuntimeError("This definitely should not happen! Invalid board: %s" % fen)

def internal_test_eval(fen):
    """
    Branch evaluation function used by Rank Prune test.
    """

    board_state, player, _, _, _, _ = fen.split(' ')
    if player != 'w':
        raise RuntimeError("This shouldn't happen! Evaluation should always be called with white next.")

    if board_state == dh.strip_fen(dh.flip_board('8/p7/1p6/8/8/PP6/8/8 w - - 0 2')):  # a2a3
        return 0.7
    elif board_state == dh.strip_fen(dh.flip_board('8/p7/1p6/8/1P6/8/P7/8 w - - 0 2')):  # b3b4
        return 0.8
    elif board_state == dh.strip_fen(dh.flip_board('8/p7/1p6/8/P7/1P6/8/8 w - - 0 2')):  # a2a4 (pruned children)
        return 0.9
    elif board_state == '8/p7/8/1p6/8/PP6/8/8':  # a2a3 -> b6b5
        return 0.0
    elif board_state == '8/8/pp6/8/8/PP6/8/8':  # a2a3 -> a7a6
        return 0.3
    elif board_state == '8/8/1p6/p7/8/PP6/8/8':  # a2a3 -> a7a5 (pruned children)
        return 0.5
    elif board_state == '8/p7/8/1p6/1P6/8/P7/8':  # b3b4 -> b6b5
        return 0.7
    elif board_state == '8/8/pp6/8/1P6/8/P7/8':  # b3b4 -> a7a6
        return 0.8
    elif board_state == '8/8/1p6/p7/1P6/8/P7/8':  # b3b4 -> a7a5 (pruned children)
        return 0.9
    elif board_state == '8/p7/1p6/8/8/1P6/P7/8':  # ROOT
        return 0.5
    else:
        raise RuntimeError("This definitely should not happen! Invalid board: %s" % board_state)


def search_timing_test(min_time=5, max_time=20, time_step=5, verbose=True):
    """
    Tests that Rank-Prune and ID searching abide by the input time limit.
    Input:
        min_time [Int] (Optional)
            Minimum search time to test.
        max_time [Int] (Optional)
            Maximum search time to test. (Inclusive)
        time_step [Int] (Optional)
            Time step size between min_time and max_time where tests are run.
        verbose [Boolean] (Optional)
            Turn Verbose mode On or Off.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    # TODO: Add varying prune_perc
    search_modes = {
        "IterativeDeepening": IterativeDeepening(evaluation_function=lambda x: random.random(), max_depth=None)}
    board = chess.Board(fen="2bq2R1/3P1PP1/3k3P/pP6/7N/NP1B3p/p1pQn2p/3nB2K w - - 0 1")  # Random FEN

    for name, search in search_modes.iteritems():
        for t in xrange(min_time, max_time + 1, time_step):
            start_time = time.time()
            search.run(board, time_limit=t)
            time_taken = time.time() - start_time
            if verbose:
                print "Timing Test: Time Limit %f. Time Taken %f." % (float(t), time_taken)
            if time_taken > t:
                print "Error in %s: time allowed: %f, time taken: %f" % (name, float(t), time_taken)
                return False
    return True


def minimax_test():
    """
    Runs a basic minimax and pruning test on the Minimax.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    # Made up starting positions with white pawns in a2 & b3 and black pawns in a7 & b6 (no kings haha)
    # This allows for only 3 nodes at depth 1, 9 nodes at depth 2, and 21 nodes at depth 3 (max)

    root_fen = "8/p7/1p6/8/8/1P6/P7/8 w - - 0 1"
    max_depth = 3

    # Built SearchNode tree based on this root_fen and the corresponding minimax_basic_eval()

    search = Minimax(minimax_test_eval, max_depth=max_depth)
    search.order_moves = False  # Turn off move ordering, use default

    try:
        score, move, fen = search.run(chess.Board(root_fen))
    except RuntimeError as e:
        print e
        return False

    if (score == 0.6) and (str(move) == "b3b4") and (fen == "8/8/pp6/8/1P6/P7/8/8 b - - 0 2"):
        return True
    else:
        print "Mimimax Prune Test Failed: Expected [Score, Move]: [0.6, b3b4, 8/8/pp6/8/1P6/P7/8/8 b - - 0 2]" \
              " got: [%.1f, %s, %s]" % (score, move, fen)
        return False


def basic_search_test(search_modes=None):
    """ Tests that search functions properly when playing white and black.
        Input:
            search_modes [String or List of Strings] (Optional)
                Can optionally specify the search modes to test. If 'None' then test all search modes.
        Output:
            Result [Boolean]
                True if test passed, False if test failed.
                """

    search_modes = [Minimax(basic_test_eval, max_depth=1),
                    IterativeDeepening(basic_test_eval, time_limit=10, max_depth=1)]

    success = True

    fens = [None] * 2
    fens[0] = "8/p7/1p6/8/8/1P6/P7/8 w - - 0 1"  # White plays next
    fens[1] = "8/p7/1p6/8/8/1P6/P7/8 b - - 0 1"  # Black plays next

    expected = [(0.6, "b3b4"), (0.6, "b6b5")]

    for i, fen_str in enumerate(fens):
        for search_mode in search_modes:
            board = chess.Board(fen=fen_str)
            # Can't run deeper due to restricted evaluation function.

            score, move, _ = search_mode.run(board)
            exp_score, exp_move = expected[i]
            if (score != exp_score) or (str(move) != exp_move):
                print "%s White Search Test Failed. Expected [Score, Move]: [0.6, b3b4] got: [%.1f, %s]" % \
                      (str(search_mode).capitalize(), score, move)
                success = False

    return success


def checkmate_search_test():
    """
    Tests that checkmates are working.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    search_modes = [Minimax((lambda x: 0.5), max_depth=1),
                    IterativeDeepening((lambda x: 0.5), time_limit=5, max_depth=2)]

    success = True

    for search_mode in search_modes:

        # Checkmates on this turn
        black_loses = chess.Board('R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1')
        white_loses = chess.Board('8/8/8/8/8/2k5/1p6/rK6 w - - 0 1')
        result, _, _ = search_mode.run(black_loses)
        if result != dh.LOSE_VALUE:
            print "%s Checkmate Search Test failed, invalid result for black checkmate." % search_mode
            success = False
        result, _, _ = search_mode.run(white_loses)
        if result != dh.LOSE_VALUE:
            print "%s Checkmate search test failed, invalid result for white checkmate." % search_mode
            success = False

        # Checkmates on next turn
        white_wins_next = chess.Board('6k1/R4ppp/8/8/8/8/8/4K3 w - - 0 1')
        black_wins_next = chess.Board('8/8/8/8/8/2k5/rp6/1K6 b - - 0 1')

        result, move, _ = search_mode.run(white_wins_next)
        if result != dh.WIN_VALUE or str(move) != 'a7a8':
            print "%s Checkmate Search test failed, invalid result for white checkmating black." % search_mode
            success = False
        result, move, _ = search_mode.run(black_wins_next)
        if result != dh.WIN_VALUE or str(move) != 'a2a1':
            print "%s Checkmate Search test failed, invalid result for black checkmating white." % search_mode
            success = False

    return success

# Used in white search, black search and checkmate search test
def basic_test_eval(fen):
    board_state, player, _, _, _, _ = fen.split(' ')
    if player != 'w':
        raise RuntimeError("This shouldn't happen! Evaluation should always be called with white next.")

    # from white plays next
    if board_state == dh.strip_fen(dh.flip_board("8/p7/1p6/8/8/PP6/8/8 w KQkq - 0 1")):  # a2a3
        return -0.5
    elif board_state == dh.strip_fen(dh.flip_board("8/p7/1p6/8/1P6/8/P7/8 w KQkq - 0 1")):  # b3b4
        return -0.6
    elif board_state == dh.strip_fen(dh.flip_board("8/p7/1p6/8/P7/1P6/8/8 w KQkq - 0 1")):  # a2a4
        return -0.3
    # from black plays next
    elif board_state == "8/8/pp6/8/8/1P6/P7/8":  # b7b6
        return 0.5
    elif board_state == "8/p7/8/1p6/8/1P6/P7/8":  # b6b5
        return 0.4
    elif board_state == "8/8/1p6/p7/8/1P6/P7/8":  # a7a5
        return 0.7
    # Root
    elif board_state == "8/p7/1p6/8/8/1P6/P7/8":
        return 0.5
    # Start board, used for timing:
    elif board_state == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR":
        return 0.5
    else:
        raise RuntimeError("This definitely should not happen! Invalid board: %s" % board_state)


# Used in minimax + pruning test
def minimax_test_eval(fen):
    """
    Arbitrary evaluation function for minimax test. Should yield move of b3b4 and value of 0.6.
    """
    board_state, player, _, _, _, _ = fen.split(' ')
    if player != 'w':
        raise RuntimeError("This shouldn't happen! Evaluation should always be called with white next.")

    board_state = dh.strip_fen(dh.flip_board(fen))

    # list of fens which should not be reached since they should be pruned
    pruned_fens = ["8/8/1p6/p7/1P6/P7/8/8",  # a2a3 -> a7a5 -> b3b4
                   "8/8/1p6/pP6/8/8/P7/8",  # b3b4 -> a7a5 -> b4b5
                   "8/8/1p6/p7/1P6/P7/8/8",  # b3b4 -> a7a5 -> a2a3
                   "8/8/1p6/p7/PP6/8/8/8",  # b3b4 -> a7a5 -> a2a4
                   "8/8/1p6/p7/PP6/8/8/8"  # a2a4 -> a7a5 ->b3b4
                   ]

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
    elif board_state == "8/p7/8/1p6/PP6/8/8/8":
        score = 0.7
    elif board_state == "8/8/pp6/1P6/8/8/P7/8":
        score = 0.1
    elif board_state == "8/8/pp6/8/PP6/8/8/8":
        score = 0.3
    elif board_state == "8/8/1p6/P7/8/8/P7/8":
        score = 0.8
    elif board_state == "8/p7/8/1P6/8/1P6/8/8":
        score = 0.99
    elif board_state == "8/p7/8/Pp6/8/1P6/8/8":
        score = 0.6
    elif board_state == "8/8/pp6/P7/8/1P6/8/8":
        score = 0.1
    # Start board, used for timing:
    elif board_state == "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR":
        score = 0.5
    elif board_state in pruned_fens:
        raise RuntimeError("Encountered FEN {} but it should have been pruned!".format(board_state))
    else:
        raise RuntimeError("This definitely should not happen! Invalid board: %s" % board_state)

    return -score


def run_play_tests():
    all_tests = {}

    all_tests["Search Tests"] = {
        'Basic Search': basic_search_test,
        'Checkmate Search': checkmate_search_test,
        'Minimax Search': minimax_test,
        # 'Iterative-Prune Search' : iterative_prune_test,
        # 'Top k-items': k_bot_test,
        # 'Partition': partition_test,
        # 'Quickselect': quickselect_test,
        # 'Search Time': search_timing_test,
    }

    all_tests["Neural Net Tests"] = {
        # 'Weight Save and Load': save_load_weights_test
    }

    success = True
    print "\nRunning Play Tests...\n"
    for group_name, group_dict in all_tests.iteritems():
        print "--- " + group_name + " ---"
        for name, test in group_dict.iteritems():
            print "Testing " + name + "..."
            if not test():
                print "%s test failed" % name.capitalize()
                success = False

    return success


def main():
    if run_play_tests():
        print "All tests passed"
    else:
        print "You broke something - go fix it"


if __name__ == '__main__':
    main()