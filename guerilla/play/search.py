import chess
import guerilla.play.data_handler as dh
import guerilla.play.node as mc_node
import time

class Search:
    """
    Implements game tree search.
    """

    def __init__(self, eval_fn, max_depth=2, search_mode="recipromax"):
        # Evaluation function must yield a score between 0 and 1.
        # Search options
        self.search_opts = {"recipromax": self.recipromax,
                            "montecarlo": self.monte_carlo}

        if search_mode not in self.search_opts:
            raise NotImplementedError("Invalid Search option!")
        self.search_mode = search_mode

        self.eval_function = eval_fn
        self.max_depth = max_depth
        self.win_value = 1
        self.lose_value = 0
        self.draw_value = 0.5
        self.reci_prune = True

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

    def monte_carlo(self, board, search_time):
        start_time = time.clock()
        while time.clock - start_time < search_time:
            # On first expansion, generate all children - not sure if this is the right way to do this but i'm tired, double check later
            if mc_node.root is None:
                mc_node(None, board.fen(), 0)
                mc_node.root.expand(-1)

            node = mc_node.select()
            new_nodes = node.expand()
            for n in new_nodes:
                n.simulate()
                n.backpropagate()

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

        # Check if draw
        if board.is_checkmate():
            return self.lose_value, None, board.fen()
        elif board.can_claim_draw() or board.is_stalemate():
            return self.draw_value, None, board.fen()
        elif depth == self.max_depth:
            fen = leaf_board = board.fen()
            if dh.black_is_next(fen):
                fen = dh.flip_board(fen)
            return self.eval_function(fen), None, leaf_board

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
                if self.reci_prune and best_score >= a:
                    break

        # print "D%d: best: %.1f, %s" % (depth, best_score, best_move)
        return best_score, best_move, best_leaf
