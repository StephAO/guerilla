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


def rank_prune_test():
    """
    Tests Rank-Prune Searching. This includes:
        (1) The correct nodes are pruned at each level.
        (2) The nodes are evaluated with the correct evaluation function (leaf VS inner).
        (3) The correct number of nodes are traversed.
        (4) The correct move is chosen.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    fen_str = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"

    board = chess.Board(fen=fen_str)
    # Can't run deeper due to restricted evaluatoin function.
    search = RankPrune(leaf_eval=minimax_test_eval, internal_eval=internal_test_eval,
                       prune_perc=0.5, max_depth=3)
    score, move, leaf_fen, root = search.run(board, time_limit=float("inf"), return_root=True)

    # Check that there are the correct number of nodes at every level and they contain the right values.
    queue = Queue.Queue()
    queue.put(root)
    depth_count = [0] * 4
    while queue.qsize() > 0:
        curr_node = queue.get()
        depth_count[curr_node.depth] += 1

        # Compare value to correct evaluation function
        fen = curr_node.fen if dh.white_is_next(curr_node.fen) else dh.flip_board(curr_node.fen)
        if curr_node.depth == 3 and minimax_test_eval(fen) != curr_node.value:
            print "Rank Prune Test Failed! Was expecting leaf value %f for %s" % (minimax_test_eval(fen), curr_node)
            return False
        elif curr_node.depth < 3 and internal_test_eval(fen) != curr_node.value:
            print "Rank Prune Test Failed! Was expecting inner value %f for %s" % (internal_test_eval(fen), curr_node)
            return False

        for child in curr_node.get_child_nodes():
            queue.put(child)

    if depth_count != [1, 3, 6, 9]:
        print "Rank Prune Test Failed! Was expecting depth counts of %s, got %s" % (str([1, 3, 6, 9]), depth_count)
        return False

    # Check result
    if (score == 0.6) and (str(move) == "b3b4") and (leaf_fen == "8/8/pp6/8/1P6/P7/8/8 b - - 0 2"):
        return True
    else:
        print "Rank Prune Test Failed: Expected [Score, Move]: [0.6, b3b4, 8/8/pp6/8/1P6/P7/8/8 b - - 0 2]" \
              " got: [%.1f, %s, %s]" % (score, move, leaf_fen)
        return False

def iterative_prune_test():
    success = True
    fen = '7k/8/5P2/8/8/8/P7/3K4 w - - 0 1'
    ip = IterativePrune(iterative_prune_test_eval, prune_perc=0.5, max_depth=2)
    score, best_move, leaf_board = ip.run(chess.Board(fen))
    if score != 0.8:
        success = False
        print "Error: Wrong score, Expected: 0.8, Actual %s" % (score)
    if str(best_move) != "f6f7":
        success = False
        print "Error: Wrong best move, Expected: f6f7, Actual %s" % (str(best_move))
    if leaf_board.split()[0] != "8/5Pk1/8/8/8/8/P7/3K4":
        success = False
        print "Error: Wrong leaf board, Expected: 8/5Pk1/8/8/8/8/P7/3K4, Actual %s" % (leaf_board.split()[0])
    curr_layer = [(None, ip.root)]
    next_layer = []
    for i in xrange(2):
        for child in curr_layer:
            move = child[0]
            node = child[1]
            fen = node.fen if node.fen.split()[1] == 'w' else dh.flip_board(node.fen)

            if fen.split()[0] =='7k/8/5P2/8/8/8/P7/3K4':
                if (node.value - 0.8) > 0.0001 or node.depth != 0 or not node.expand:
                    success = False
                    print "Error: Node with fen 7k/8/5P2/8/8/8/P7/3K4 incorrect"

            elif fen.split()[0] =='2k5/p7/8/8/8/5p2/8/7K':
                if (node.value - 0.6) > 0.0001 or node.depth != 1 or node.expand:
                    success = False
                    print "Error: Node with fen 2k5/p7/8/8/8/5p2/8/7K incorrect"

            elif fen.split()[0] =='4k3/p7/8/8/8/5p2/8/7K':
                if (node.value - 0.6) > 0.0001 or node.depth != 1 or node.expand:
                    success = False
                    print "Error: Node with fen 4k3/p7/8/8/8/5p2/8/7K incorrect"

            elif fen.split()[0] =='8/p1k5/8/8/8/5p2/8/7K':
                if (node.value - 0.6) > 0.0001 or node.depth != 1 or node.expand:
                    success = False
                    print "Error: Node with fen 8/p2k4/8/8/8/5p2/8/7K incorrect"

            elif fen.split()[0] =='8/p2k4/8/8/8/5p2/8/7K':
                if (node.value - 0.6) > 0.0001 or node.depth != 1 or node.expand:
                    success = False
                    print "Error: Node with fen 8/p1k5/8/8/8/5p2/8/7K incorrect"

            elif fen.split()[0] =='8/p3k3/8/8/8/5p2/8/7K':
                if (node.value - 0.7) > 0.0001 or node.depth != 1 or not node.expand:
                    success = False
                    print "Error: Node with fen 8/p3k3/8/8/8/5p2/8/7K is incorrect"

            elif fen.split()[0] =='3k4/8/p7/8/8/5p2/8/7K':
                if (node.value - 0.7) > 0.0001 or node.depth != 1 or not node.expand:
                    success = False
                    print "Error: Node with fen 3k4/8/p7/8/8/5p2/8/7K is incorrect"

            elif fen.split()[0] =='3k4/p7/8/8/8/8/5p2/7K':
                if (node.value - 0.2) > 0.0001 or node.depth != 1 or not node.expand:
                    success = False
                    print "Error: Node with fen 3k4/p7/8/8/8/8/5p2/7K is incorrect"
                    print float(node.value) != float(0.2), node.depth != 1, not node.expand
                    print repr(node.value), repr(0.2)

            elif fen.split()[0] =='3k4/8/8/p7/8/5p2/8/7K':
                if (node.value - 0.7) > 0.0001 or node.depth != 1 or not node.expand:
                    success = False
                    print "Error: Node with fen 3k4/8/8/p7/8/5p2/8/7K is incorrect"

            elif move =='h8g8':
                if (node.value - 0.3) > 0.0001 or node.depth != 2 or node.expand:
                    success = False
                    print "Error: Node with fen 8/7k/5P2/8/8/8/P7/2K5 is incorrect"

            elif move =='h8h7':
                if (node.value - 0.9) > 0.0001 or node.depth != 2 or not node.expand:
                    success = False
                    print "Error: Node with fen 6k1/8/5P2/8/8/8/P7/2K5 is incorrect"

            elif move == 'h8g7':
                if (node.value - 0.8) > 0.0001 or node.depth != 2 or not node.expand:
                    success = False
                    print "Error: Node with fen 8/5Pk1/8/8/8/8/P7/3K4 is incorrect"
            else:
                success = False
                print "Error: Node with fen %s shouldn't exists" % (fen)

            if node.expand:
                next_layer.extend(node.children.items())
        curr_layer = list(next_layer)
        next_layer = []

    return success

def iterative_prune_test_eval(fen):

    if fen.split()[0] == '7k/8/5P2/8/8/8/P7/3K4':
        """
        Root
        . . . . . . . k
        . . . . . . . .
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . . . . . .
        . . . K . . . .
        """
        return 0.0

    elif fen.split()[0] == '2k5/p7/8/8/8/5p2/8/7K':
        """
        Depth 1, Child of 0
        . . . . . . . k
        . . . . . . . .
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . . . . . .
        . . K . . . . .
        """
        return 0.6

    elif fen.split()[0] == '4k3/p7/8/8/8/5p2/8/7K':
        """
        Depth 1, Child of 0
        . . . . . . . k
        . . . . . . . .
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . . . . . .
        . . . . K . . .
        """
        return 0.6

    elif fen.split()[0] == '8/p1k5/8/8/8/5p2/8/7K':
        """
        Depth 1, Child of 0
        . . . . . . . k
        . . . . . . . .
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . K . . . . .
        . . . . . . . .
        """
        return 0.6

    elif fen.split()[0] == '8/p2k4/8/8/8/5p2/8/7K':
        """
        Depth 1, Child of 0
        . . . . . . . k
        . . . . . . . .
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . K . . . .
        . . . . . . . .
        """
        return 0.6

    elif fen.split()[0] == '8/p3k3/8/8/8/5p2/8/7K':
        """
        Depth 1, Child of 0
        . . . . . . . k
        . . . . . . . .
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . . K . . .
        . . . . . . . .
        """
        return 0.4

    elif fen.split()[0] == '3k4/8/p7/8/8/5p2/8/7K':
        """
        Depth 1, Child of 0
        . . . . . . . k
        . . . . . . . .
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        P . . . . . . .
        . . . . . . . .
        . . . K . . . .
        """
        return 0.4

    elif fen.split()[0] == '3k4/p7/8/8/8/8/5p2/7K':
        """
        Depth 1, Child of 0
        . . . . . . . k
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . . . . . .
        . . . K . . . .
        """
        return 0.4

    elif fen.split()[0] == '3k4/8/8/p7/8/5p2/8/7K':
        """
        Depth 1, Child of 0
        . . . . . . . k
        . . . . . . . .
        . . . . . P . .
        . . . . . . . .
        P . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . . K . . . .
        """
        return 0.4

    elif fen.split()[0][:8] == '8/7k/5P2':
        """
        Depth 2, Child of 0
        . . . . . . . .
        . . . . . . . k
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . . . . . .
        . . K . . . . .
        """
        return 0.8

    elif fen.split()[0][:9] == '6k1/8/5P2':
        """
        Depth 2, Child of 0
        . . . . . . k .
        . . . . . . . .
        . . . . . P . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . . . . . .
        . . K . . . . .
        """
        return 0.3

    elif fen.split()[0][:8] == '8/5Pk1/8':
        """
        Depth 2, Child of 6
        . . . . . . . .
        . . . . . P k .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . . . . . .
        . . . K . . . .
        """
        return 0.8

    elif fen.split()[0][:8] == '8/5P1k/8':
        """
        Depth 2, Child of 6
        . . . . . . . .
        . . . . . P . k
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        . . . . . . . .
        P . . . . . . .
        . . . K . . . .
        """
        return 0.9

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


def search_timing_test(min_time=5, max_time=20, time_step=5, verbose=False):
    """
    Tests that Rank-Prune searching abides by the input time limit.
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
    search = RankPrune(leaf_eval=lambda x: random.random(), prune_perc=0.9, buff_time=3)
    board = chess.Board(fen="2bq2R1/3P1PP1/3k3P/pP6/7N/NP1B3p/p1pQn2p/3nB2K w - - 0 1")  # Random FEN

    for t in xrange(min_time, max_time + 1, time_step):
        start_time = time.time()
        search.run(board, time_limit=t)
        time_taken = time.time() - start_time
        if verbose:
            print "Timing Test: Time Limit %f. Time Taken %f." % (float(t), time_taken)
        if time_taken > t:
            print "Error: time allowed: %f, time taken: %f" % (float(t), time_taken)
            return False
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

    pruned_flipped = map(lambda x: dh.strip_fen(dh.flip_board(x + ' w ---- - 0 1')), pruned_fens)

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
                new_node.value = minimax_test_eval(fen) if dh.strip_fen(fen) not in pruned_flipped else 0
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

    search_modes = [Complementmax(basic_test_eval, max_depth=1),
                    RankPrune(basic_test_eval, prune_perc=0, time_limit=10, max_depth=1)]

    success = True

    fens = [None] * 2
    fens[0] = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"  # White plays next
    fens[1] = "8/p7/1p6/8/8/1P6/P7/8 b ---- - 0 1"  # Black plays next

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
    Input:
        search_modes [String or List of Strings] (Optional)
            Can optionally specify the search modes to test. If 'None' then test all search modes.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    search_modes = [Complementmax((lambda x: 0.5), max_depth=1),
                    RankPrune((lambda x: 0.5), prune_perc=0, time_limit=10, max_depth=2)]

    success = True

    for search_mode in search_modes:

        # Checkmates on this turn
        black_loses = chess.Board('R5k1/5ppp/8/8/8/8/8/4K3 b - - 0 1')
        white_loses = chess.Board('8/8/8/8/8/2k5/1p6/rK6 w - - 0 1')
        result, _, _ = search_mode.run(black_loses)
        if result != 0:
            print "%s Checkmate Search Test failed, invalid result for black checkmate." % search_mode
            success = False
        result, _, _ = search_mode.run(white_loses)
        if result != 0:
            print "%s Checkmate search test failed, invalid result for white checkmate." % search_mode
            success = False

        # Checkmates on next turn
        white_wins_next = chess.Board('6k1/R4ppp/8/8/8/8/8/4K3 w - - 0 1')
        black_wins_next = chess.Board('8/8/8/8/8/2k5/rp6/1K6 b - - 0 1')

        result, move, _ = search_mode.run(white_wins_next)
        if result != 1 or str(move) != 'a7a8':
            print "%s Checkmate Search test failed, invalid result for white checkmating black." % search_mode
            success = False
        result, move, _ = search_mode.run(black_wins_next)
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
    shallow = Complementmax(minimax_test_eval, max_depth=3)
    score, move, fen = shallow.run(board)
    if (score == 0.6) and (str(move) == "b3b4"):
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
    if board_state == dh.strip_fen(dh.flip_board("8/p7/1p6/8/8/PP6/8/8 w KQkq - 0 1")):  # a2a3
        return 0.5
    elif board_state == dh.strip_fen(dh.flip_board("8/p7/1p6/8/1P6/8/P7/8 w KQkq - 0 1")):  # b3b4
        return 0.4
    elif board_state == dh.strip_fen(dh.flip_board("8/p7/1p6/8/P7/1P6/8/8 w KQkq - 0 1")):  # a2a4
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
        'Iterative-Prune Search' : iterative_prune_test,
        'Top k-items': k_bot_test,
        'Partition': partition_test,
        'Quickselect': quickselect_test,
        'MinimaxTree': minimaxtree_test,
        'Search Time': search_timing_test,
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