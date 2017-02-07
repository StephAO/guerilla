# Play Unit tests
import os
import sys

import chess
import numpy as np
from pkg_resources import resource_filename

import guerilla.data_handler as dh
import guerilla.play.neural_net as nn
from guerilla.play.search import *

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
    result_msg = dh.diff_dict_helper(weights, new_weights)
    if result_msg:
        print "Weight did not match."
        print result_msg
        return False

    return True

###############################################################################
# SEARCH TESTS
###############################################################################

def k_top_test():
    """
    Tests the k-top function used in Rank-Prune searching.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    test_list = [10, 79, 9, 59, 9, 47, 50, 41, 36, 80, 63, 25, 76, 81, 81, 30, 79, 81, 26, 52]
    top_1 = [81]
    top_3 = [81, 81, 81]
    top_5 = [81, 81, 81, 80, 79]
    top_10 = [81, 81, 81, 80, 79, 79, 76, 63, 59, 52]

    test_solutions = {
        1 : top_1, 
        3 : top_3, 
        5 : top_5, 
        10: top_10
    }

    success = True
    for test, solution in test_solutions.iteritems():
        result = k_top(list(test_list), test)
        if len(result) != test:
            print "Error: k_top does not return the correct number of items.\n" \
                  "Expected: %d, Actual: %d" % (test, len(result))
            success = False
        for item in result:
            if item not in solution:
                print "Error: k_top does not return the maximum values\n" \
                      "Expected: %s, Actual:%s" % (str(solution), str(result))
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


def rank_prune_test():
    """
    Tests Rank-Prune Searching. This includes:
        (1) The correct number of nodes is pruned at each level.
        (2) The correct tree is generated.
        (3) The nodes are evaluated with the correct evaluation function (leaf VS inner).
        (4) The correct move is chosen.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    # TODO
    return True


def search_timing_test():
    # TODO
    return True


def minimaxtree_test():
    """
    Tests the minimaxtree function. Tests that correct output is generated and pruning is done.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    root_fen = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"
    max_depth = 3

    # list of fens which should not be reached since they should be pruned
    pruned_fens = ["8/8/1p6/p7/1P6/P7/8/8",  # a2a3 -> a7a5 -> b3b4
                   "8/8/1p6/pP6/8/8/P7/8",  # b3b4 -> a7a5 -> b4b5
                   "8/8/1p6/p7/1P6/P7/8/8",  # b3b4 -> a7a5 -> a2a3
                   "8/8/1p6/p7/PP6/8/8/8",  # b3b4 -> a7a5 -> a2a4
                   "8/8/1p6/p7/PP6/8/8/8"  # a2a4 -> a7a5 ->b3b4
                   ]

    pruned_flipped = map(lambda x: dh.flip_board(x + ' w ---- - 0 1').split(' ')[0], pruned_fens)

    # Built SearchNode tree based on this root_fen and the corresponding minimax_basic_eval()

    root = SearchNode(root_fen, 0, random.random())

    depth = 1
    stack = [root]
    while stack != []:
        curr_node = stack.pop()

        board = chess.Board(curr_node.fen)
        for move in board.legal_moves:
            board.push(move)
            new_node = SearchNode(board.fen(), curr_node.depth + 1, random.random())

            fen = new_node.fen
            if dh.black_is_next(fen):
                fen = dh.flip_board(fen)

            # Check if max depth
            if new_node.depth == max_depth:
                new_node.value = minimax_test_eval(fen) if fen.split(' ')[0] not in pruned_flipped else 0
            else:
                stack.append(new_node)

            curr_node.add_child(move, new_node)
            board.pop()

    try:
        score, move, fen = minimaxtree(root, forbidden_fens=pruned_fens)
    except RuntimeError as e:
        print e
        return False

    if (score == 0.6) and (str(move) == "b3b4") and (fen == "8/8/pp6/8/1P6/P7/8/8 b - - 0 2"):
        return True
    else:
        print "Mimimaxree Test Failed: Expected [Score, Move]: [6, b3b4, 8/8/pp6/8/1P6/P7/8/8 b - - 0 2]" \
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

    if search_modes is None:
        search_modes = Search.search_modes
    elif not isinstance(search_modes, list):
        search_modes = list(search_modes)

    success = True

    fens = [None] * 2
    fens[0] = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"  # White plays next
    fens[1] = "8/p7/1p6/8/8/1P6/P7/8 b ---- - 0 1"  # Black plays next

    expected = [(0.6, "b3b4"), (0.6, "b6b5")]

    for i, fen_str in enumerate(fens):
        for search_mode in search_modes:
            board = chess.Board(fen=fen_str)
            # Can't run deeper due to restricted evaluation function.
            shallow = Search(basic_test_eval, max_depth=1, search_mode=search_mode)

            # Set Rank Prune parameters
            shallow.prune_perc = 0
            shallow.time_limit = 10
            shallow.limit_depth = True

            score, move, _ = shallow.run(board)
            exp_score, exp_move = expected[i]
            if (score != exp_score) or (str(move) != exp_move):
                print "%s White Search Test Failed. Expected [Score, Move]: [0.6, b3b4] got: [%.1f, %s]" % \
                      (search_mode.capitalize(), score, move)
                success = False

    return success


def checkmate_search_test(search_modes=None):
    """
    Tests that checkmates are working.
    Input:
        search_modes [String or List of Strings] (Optional)
            Can optionally specify the search modes to test. If 'None' then test all search modes.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    if search_modes is None:
        search_modes = Search.search_modes
    elif not isinstance(search_modes, list):
        search_modes = list(search_modes)

    success = True

    for search_mode in search_modes:
        s = Search((lambda x: 0.5), max_depth=1, search_mode=search_mode)

        # Set Rank Prune parameters
        s.prune_perc = 0
        s.time_limit = 10
        s.limit_depth = True

        # Checkmates on this turn
        black_loses = chess.Board('R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1')
        white_loses = chess.Board('8/8/8/8/8/2k5/1p6/rK6 w - - 0 1')
        result, _, _ = s.run(black_loses)
        if result != 0:
            print "%s Checkmate Search Test failed, invalid result for black checkmate." % search_mode
            success = False
        result, _, _ = s.run(white_loses)
        if result != 0:
            print "%s Checkmate search test failed, invalid result for white checkmate." % search_mode
            success = False

        # Checkmates on next turn
        white_wins_next = chess.Board('6k1/R4ppp/8/8/8/8/8/4K3 w - - 0 1')
        black_wins_next = chess.Board('8/8/8/8/8/2k5/rp6/1K6 b - - 0 1')

        result, move, _ = s.run(white_wins_next)
        if result != 1 or str(move) != 'a7a8':
            print "%s Checkmate Search test failed, invalid result for white checkmating black." % search_mode
            success = False
        result, move, _ = s.run(black_wins_next)
        if result != 1 or str(move) != 'a2a1':
            print "%s Checkmate Search test failed, invalid result for black checkmating white." % search_mode
            success = False

    return success


def complementmax_test():
    """ Runs a basic minimax and pruning test on the Complemenetmax. """

    # Made up starting positions with white pawns in a2 & b3 and black pawns in a7 & b6 (no kings haha)
    # This allows for only 3 nodes at depth 1, 9 nodes at depth 2, and 21 nodes at depth 3 (max)

    fen_str = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"

    board = chess.Board(fen=fen_str)
    # Can't run deeper due to restricted evaluatoin function.
    shallow = Search(minimax_test_eval, max_depth=3, search_mode='complementmax')
    score, move, fen = shallow.run(board)
    if (score == 0.6) and (str(move) == "b3b4") and (fen == "8/8/pp6/8/1P6/P7/8/8 b - - 0 2"):
        return True
    else:
        print "ComplementMax Test Failed: Expected [Score, Move]: [6, b3b4, 8/8/pp6/8/1P6/P7/8/8 b - - 0 2]" \
              " got: [%.1f, %s, %s]" % (score, move, fen)
        return False


# Used in white search, black search and checkmate search test
def basic_test_eval(fen):
    board_state, player, _, _, _, _ = fen.split(' ')
    if player != 'w':
        raise RuntimeError("This shouldn't happen! Evaluation should always be called with white next.")

    # from white plays next
    if board_state == dh.flip_board("8/p7/1p6/8/8/PP6/8/8 w KQkq - 0 1").split(' ')[0]:  # a2a3
        return 0.5
    elif board_state == dh.flip_board("8/p7/1p6/8/1P6/8/P7/8 w KQkq - 0 1").split(' ')[0]:  # b3b4
        return 0.4
    elif board_state == dh.flip_board("8/p7/1p6/8/P7/1P6/8/8 w KQkq - 0 1").split(' ')[0]:  # a2a4
        return 0.7
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
    else:
        raise RuntimeError("This definitely should not happen! Invalid board: %s" % board_state)


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
    else:
        raise RuntimeError("This definitely should not happen! Invalid board: %s" % board_state)

    return 1 - score

def run_play_tests():
    all_tests = {}

    all_tests["Search Tests"] = {
        'Basic Search': basic_search_test,
        'Checkmate Search': checkmate_search_test,
        'Complementmax Search': complementmax_test,
        'Rank-Prune Search': rank_prune_test,
        'Top k-items': k_top_test,
        'Partition': partition_test,
        'Quickselect': quickselect_test,
        'MinimaxTree': minimaxtree_test
        }

    all_tests["Neural Net Tests"] = {
        'Weight Save and Load': save_load_weights_test
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