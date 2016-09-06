import chess

class Search:
    max_depth = 3

    def __init__(self, eval_fn):
        self.eval_function = eval_fn

    def minimax(board, depth, a, B, max_player):
        """ 
            Recursive function to search for best move using minimax with alpha-beta pruning
            This has been tested
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
        if depth == max_depth:
            return self.eval_function(board), None
        else:
            for move in board.legal_moves:
                # recursive call
                board.push(move)
                score, next_move = minimax(board, depth+1, a, B, not max_player)
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
        return best_score, best_move


board = chess.Board(chess.STARTING_FEN)