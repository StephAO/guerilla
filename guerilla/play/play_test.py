# Play Unit tests
import os

import chess
import numpy as np
from pkg_resources import resource_filename

import guerilla.data_handler as dh
import guerilla.play.neural_net as nn
from guerilla.play.search import Search

###############################################################################
# Input Tests
###############################################################################

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
        actual = dh.get_diagonals(dh.fen_to_bitmap(fens[i]))
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
        actual = dh.fen_to_bitmap(fens[i])
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

def run_play_tests():
    all_tests = {}
    all_tests["Input Tests"] = {'Board to Channels': channel_input_test,
                                'Channels to Diagonals': diag_input_test}

    all_tests["Search Tests"] = {'White Search': white_search_test,
                                 'Black Search': black_search_test,
                                 'Checkmate Search': checkmate_search_test,
                                 'Minimax And Pruning': minimax_pruning_test}

    all_tests["Neural Net Tests"] = {'Weight Save and Load': save_load_weights_test}


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