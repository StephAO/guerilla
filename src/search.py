import chess
import data_handler as dh


class Search:
    """
    Implements game tree search.
    """

    def __init__(self, eval_fn, max_depth=3, search_mode="recipromax"):
        # Evaluation function must yield a score between 0 and 1.
        # Search options
        self.search_opts = {"recipromax": self.recipromax}

        if search_mode not in self.search_opts:
            raise NotImplementedError("Invalid Search option!")
        self.search_mode = search_mode

        self.eval_function = eval_fn
        self.max_depth = max_depth

    def run(self, board):
        """
        Runs search based on parameter.
        Inputs:
                board [chess.Board]:
                    current state of board
            Outputs:
                best_move [chess.Move]:
                    Best move to play
                best_score [float]:
                    Score achieved by best move
                best_leaf [String]
                    FEN of the board of the leaf node which yielded the highest value.
        """

        return self.search_opts[self.search_mode](board)

    def recipromax(self, board, depth=0, a=1.0):
        """ 
            Recursive function to search for best move using recipromax with alpha-beta pruning.
            Assumes that the layer above the leaves are trying to minimize the positive value,
            which is the same as maximizing the reciprocal.
            Inputs:
                board [chess.Board]:
                    current state of board
                depth [int]:
                    current depth, used for terminating condition
                a [float]:
                    lower bound of layer above, upper bound of current layer (because of alternating signs)
            Outputs:
                best_score [float]:
                    Score achieved by best move
                best_move [chess.Move]:
                    Best move to play
                best_leaf [String]
                    FEN of the board of the leaf node which yielded the highest value.
        """
        best_score = 0.0
        best_move = None
        best_leaf = None

        if depth == self.max_depth:
            fen = leaf_board = board.fen()
            if dh.black_is_next(fen):
                fen = dh.flip_board(fen)
            return self.eval_function(fen), None, leaf_board

        ##### If using search_test2() ######
        # if type(board) is int:
        #     return (-1)*board, None
        ####################################

        else:
            for move in board.legal_moves:
                # print "D%d: %s" % (depth, move)
                # recursive call
                board.push(move)
                # print move
                score, next_move, leaf_board = self.recipromax(board, depth + 1, 1 - best_score)
                # Take reciprocal of score since alternating levels
                score = 1 - score
                board.pop()
                # print "D: %d M: %s S: %.1f" % (depth, move, score)
                if score >= best_score:
                    best_score = score
                    best_move = move
                    best_leaf = leaf_board

                # best_score is my current lower bound
                # a is the upper bound of what's useful to search
                # if my lower bound breaks the boundaries of what's worth to search
                # stop searching here
                if best_score >= a:
                    break

                    ##### If using search_test2() ######
                    # end = False
                    # for sub in board:
                    #     if end:
                    #         print "cut out", sub
                    #     else:
                    #         score, next_move = self.recipromax(sub, depth+1, best_score)
                    #         if score > best_score:
                    #             best_score = score
                    #             best_move = sub
                    #         if best_score >= (-1)*a:
                    #             print "D%d: %s, %s" % (depth, best_score, best_move)
                    #             end = True
                    ###################################

        # print "D%d: best: %.1f, %s" % (depth, best_score, best_move)
        return best_score, best_move, best_leaf


def search_test_eval(fen):
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
        print "WARNING: This node should not be reached when using alpha-beta pruning!"
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
        print "WARNING: This node should not be reached when using alpha-beta pruning!"
        score = 0.0
    elif board_state == "8/8/1p6/p7/PP6/8/8/8":
        print "WARNING: This node should not be reached when using alpha-beta pruning!"
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


def search_test_eval2(fen):
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


def search_test1():
    """ Runs a basic minimax test on the search class. """

    # Made up starting positions with white pawns in a2 & b3 and black pawns in a7 & b6 (no kings haha)
    # This allows for only 3 nodes at depth 1, 9 nodes at depth 2, and 21 nodes at depth 3 (max)

    fen_str = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"

    board = chess.Board(fen=fen_str)
    # Can't run deeper due to restricted evaluatoin function.
    shallow = Search(eval_fn=search_test_eval, max_depth=3, search_mode='recipromax')
    score, move, _ = shallow.run(board)
    if (score == 0.6) and (str(move) == "b3b4"):
        print "Test 1 passed."
        return True
    else:
        print "Test 1 failed. Expected [Score, Move]: [6, b3b4] got: [%.1f, %s]" % (score, move)
        return False


def search_test2():
    fen_str = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"

    board = chess.Board(fen=fen_str)
    # Can't run deeper due to restricted evaluatoin function.
    shallow = Search(eval_fn=search_test_eval2, max_depth=1, search_mode='recipromax')
    score, move, _ = shallow.run(board)
    if (score == 0.6) and (str(move) == "b3b4"):
        print "Test 2 passed."
        return True
    else:
        print "Test 2 failed. Expected [Score, Move]: [0.6, b3b4] got: [%.1f, %s]" % (score, move)
        return False


def search_test3():
    fen_str = dh.flip_board("8/p7/1p6/8/8/1P6/P7/8 b ---- - 0 1")

    board = chess.Board(fen=fen_str)
    # Can't run deeper due to restricted evaluatoin function.
    shallow = Search(eval_fn=search_test_eval2, max_depth=1, search_mode='recipromax')
    score, move, _ = shallow.run(board)
    if (score == 0.6) and (str(move) == "b3b4"):
        print "Test 3 passed."
        return True
    else:
        print "Test 3 failed. Expected [Score, Move]: [0.6, b3b4] got: [%.1f, %s]" % (score, move)
        return False


def search_test_old():
    # DEPRECATED
    """ Runs a basic minimax test on the search class. 
        You need to toggle the comments in recipromax to test it
        You can generate more tests at http://inst.eecs.berkeley.edu/~cs61b/fa14/ta-materials/apps/ab_tree_practice/
        Ensure that the layer above the leaves are minimum layers"""
    test = [[[-15, -15, 2], [4, 9, 4], [-16, 3, -15]], [[18, 7, -2], [10, -6, 15], [-2, -15, -13]],
            [[12, 16, -19], [6, 14, -10], [2, -6, -7]]]
    s = Search(None)
    result = s.recipromax(test)
    print result
    if result == (-7, [[12, 16, -19], [6, 14, -10], [2, -6, -7]]):
        print "Test passed."
        return True
    else:
        print "Test failed. Expected (-7, [[12, 16, -19], [6, 14, -10], [2, -6, -7]]) got %s" % result


if __name__ == '__main__':
    search_test1()
    search_test2()
    search_test3()
