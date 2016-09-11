import chess

class Search:
    """
    Implements game tree search.
    """
    def __init__(self, eval_fn, max_depth=3):
        self.eval_function = eval_fn
        self.max_depth = max_depth

    def negamax(self, board, depth=0, a=float("-inf")):
        """ 
            Recursive function to search for best move using negamax with alpha-beta pruning.
            Assumes that the layer above the leaves are trying to minimize the positive value,
            which is the same as maximizing the negatives.
            Inputs:
                board [chess.Board]:
                    current state of board
                depth [int]:
                    current depth, used for terminating condition
                a [float]:
                    lower bound of layer above, upper bound of current layer (because of alternating signs)
            Outputs:
                best_move [chess.Move]:
                    Best move to play
                best_score [float]:
                    Score achieved by best move
        """
        best_score = float("-inf")
        best_move = None

        if depth == self.max_depth:
            fen = board.fen()
            if fen[1] == 'b':
                fen = dc.flip_board(fen)
            return (-1)*self.eval_function(board), None
            
        ##### If using search_test2() ######
        # if type(board) is int:
        #     return (-1)*board, None
        ####################################
        
        else:
            for move in board.legal_moves:
                # print "D%d: %s" % (depth, move)
                # recursive call
                board.push(move)
                score, next_move = self.negamax(board, depth+1, -best_score)
                board.pop()

                if score > best_score:
                    best_score = score
                    best_move = move

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
            #         score, next_move = self.negamax(sub, depth+1, best_score)
            #         if score > best_score:
            #             best_score = score
            #             best_move = sub
            #         if best_score >= (-1)*a:
            #             print "D%d: %s, %s" % (depth, best_score, best_move)
            #             end = True
            ###################################
            
        # print "D%d: best: %s, %s" % (depth, best_score, best_move)
        return (-1)*best_score, best_move

def search_test_eval(board):
    """
    Arbitrary evaluation function for minimax test. Should yield move of b3b4 and value of 6.
    """
    board_state = board.fen().split(' ')[0]

    if board_state == "8/p7/8/1p6/P7/1P6/8/8":
        return 5
    elif board_state == "8/p7/8/1p6/1P6/P7/8/8":
        return 4
    elif board_state == "8/8/pp6/8/P7/1P6/8/8":
        return 1
    elif board_state == "8/8/pp6/8/1P6/P7/8/8":
        return 6
    elif board_state == "8/8/1p6/p7/P7/1P6/8/8":
        return 8
    elif board_state == "8/8/1p6/p7/1P6/P7/8/8":
        print "WARNING: This node should not be reached when using alpha-beta pruning!"
        return 0
    elif board_state == "8/p7/8/1p6/PP6/8/8/8":
        return 7
    elif board_state == "8/8/pp6/1P6/8/8/P7/8":
        return 1
    elif board_state == "8/8/pp6/8/PP6/8/8/8":
        return 3
    elif board_state == "8/8/1p6/P7/8/8/P7/8":
        return 8
    elif board_state == "8/8/1p6/pP6/8/8/P7/8":
        print "WARNING: This node should not be reached when using alpha-beta pruning!"
        return 0
    elif board_state == "8/8/1p6/p7/PP6/8/8/8":
        print "WARNING: This node should not be reached when using alpha-beta pruning!"
        return 0
    elif board_state == "8/p7/8/1P6/8/1P6/8/8":
        return 10
    elif board_state == "8/p7/8/Pp6/8/1P6/8/8":
        return 6
    elif board_state == "8/8/pp6/P7/8/1P6/8/8":
        return 1
    else:
        raise RuntimeError("This definitely should not happen! Invalid board.")


def search_test1():
    """ Runs a basic minimax test on the search class. """

    # Made up starting positions with white pawns in a2 & b3 and black pawns in a7 & b6 (no kings haha)
    # This allows for only 3 nodes at depth 1, 9 nodes at depth 2, and 21 nodes at depth 3 (max)

    fen_str = "8/p7/1p6/8/8/1P6/P7/8 w ---- - 0 1"

    board = chess.Board(fen=fen_str)
    shallow = Search(eval_fn=search_test_eval, max_depth=3)  # Can't run deeper due to restricted evaluatoin function.
    score, move = shallow.minimax(board)
    if (score == 6) and (str(move) == "b3b4"):
        print "Test passed."
        return True
    else:
        print "Test failed. Expected [Score, Move]: [6, b3b4] got [%d, %s]" % (score, move)
        return False

def search_test2():
    """ Runs a basic minimax test on the search class. 
        You need to toggle the comments in negamax to test it
        You can generate more tests at http://inst.eecs.berkeley.edu/~cs61b/fa14/ta-materials/apps/ab_tree_practice/
        Ensure that the layer above the leaves are minimum layers"""
    test = [[[-15,-15,2],[4,9,4],[-16,3,-15]],[[18,7,-2],[10,-6,15],[-2,-15,-13]],[[12,16,-19],[6,14,-10],[2,-6,-7]]]
    s = Search(None)
    result = s.negamax(test)
    print result
    if result == (-7, [[12, 16, -19], [6, 14, -10], [2, -6, -7]]):
        print "Test passed."
        return True
    else:
        print "Test failed. Expected (-7, [[12, 16, -19], [6, 14, -10], [2, -6, -7]]) got %s" % (result)


if __name__ == '__main__':
    search_test2()
