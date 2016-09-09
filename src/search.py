import chess


class Search:
    """
    Implements game tree search.
    """

    def __init__(self, eval_fn, max_depth=3):
        self.eval_function = eval_fn
        self.max_depth = max_depth

    def minimax(self, board, depth=0, a=float("-inf"), B=float("inf"), max_player=True):
        """ 
            Recursive function to search for best move using minimax with alpha-beta pruning.
            Tested.
            Inputs:
                board [chess.Board]:
                    current state of board
                depth [int]:
                    current depth, used for terminating condition
                a [float]:
                    max player lower bound. Alpha part of alpha-beta pruning
                B [float]:
                    min player upper bound. Beta part of alpha-beta pruning
                max_player [bool]:
                    boolean to decided whether to maximize or minimize search value
            Outputs:
                best_move [chess.Move]:
                    Best move to play
                best_score [float]:
                    Score achieved by best move
        """
        best_score = float("-inf") if max_player else float("inf")
        best_move = None
        if depth == self.max_depth:
            return self.eval_function(board), None
        else:
            for move in board.legal_moves:
                # print "D%d: %s" % (depth, move)
                # recursive call
                board.push(move)
                score, next_move = self.minimax(board, depth + 1, a, B, not max_player)
                board.pop()
                if max_player:
                    if score > a:
                        a = score
                    if score > best_score:
                        best_score = score
                        best_move = move
                else:
                    if score < B:
                        B = score
                    if score < best_score:
                        best_score = score
                        best_move = move
                if B <= a:
                    break
        # print "D%d: best: %s, %s" % (depth, best_score, best_move)
        return best_score, best_move


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


def search_test():
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
