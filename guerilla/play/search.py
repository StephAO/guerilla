from abc import ABCMeta, abstractmethod, abstractproperty
import time
import Queue
import chess
import random
import math

import guerilla.data_handler as dh


class Search:
    __metaclass__ = ABCMeta

    def __init__(self, evaluation_function):
        """
        Initializes the Player abstract class.
        Input:
            leaf_evaluation_function [function]:
                function to evaluate leaf nodes (usually the neural net)
        """
        self.evaluation_function = evaluation_function
        self.win_value = 1
        self.lose_value = 0
        self.draw_value = 0.5
        self.cache_hits = 0
        self.cache_miss = 0
        # cache: Key is FEN, Value is Score
        self.cache = {}

    @abstractmethod
    def condition(self, depth):
        # TODO: Maybe consider making this more general? For example what if depth isn't the only condition in the future. Maybe use **kwargs
        """
        Set condition for determining whether a given node should be evaluated.
        Returns True if the node should be evaluated.
        Inputs:
            board[chess.Board]:
                current state of the board
        Output:
            [Boolean]
                Whether the board should be evaluated or not.
        """
        raise NotImplementedError("You should never see this")

    @abstractmethod
    def run(self, board, time_limit=None, clear_cache=False):
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
        raise NotImplementedError("You should never see this")

    def check_conditions(self, board, depth):
        # TODO: Function description
        fen = unflipped_fen = board.fen()
        if dh.black_is_next(fen):
            fen = dh.flip_board(fen)
        # Check if draw
        if board.is_checkmate():
            return self.lose_value, None, unflipped_fen
        elif board.can_claim_draw() or board.is_stalemate():
            return self.draw_value, None, unflipped_fen
        elif self.condition(depth):
            # Check for self.cache hit
            if fen not in self.cache:
                self.cache_hits += 1
                self.cache[fen] = self.evaluation_function(fen)
            else:
                self.cache_miss += 1

            return self.cache[fen], None, unflipped_fen
        else:
            return None

    @abstractmethod
    def __str__(self):
        raise NotImplementedError("You should never see this")


class Complementmax(Search):
    """
    Implements game tree search.
    """

    def __init__(self, leaf_eval, max_depth=2):
        # Evaluation function must yield a score between 0 and 1.
        # Search options
        super(Complementmax, self).__init__(leaf_eval)

        self.evaluation_function = leaf_eval
        self.max_depth = max_depth
        self.reci_prune = True

    def condition(self, depth):
        return depth == self.max_depth

    def run(self, board, time_limit=None, clear_cache=False):
        if clear_cache:
            self.cache = {}
        return self.complementmax(board)

    def complementmax(self, board, depth=0, a=1.0):
        """ 
            Recursive function to search for best move using complementomax with alpha-beta pruning.
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

        end = self.check_conditions(board, depth)
        if end is not None:
            return end
        else:
            for move in board.legal_moves:
                # print "D%d: %s" % (depth, move)
                # recursive call
                board.push(move)
                # print move
                score, next_move, leaf_board = self.complementmax(board, depth + 1, 1 - best_score)
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

    def __str__(self):
        return "Complementmax"


# RANK PRUNE
class RankPrune(Search):
    def __init__(self, leaf_eval, branch_eval=None, prune_perc=0.5,
                 time_limit=10, buff_time=3, limit_depth=False, max_depth=3):
        # Evaluation function must yield a score between 0 and 1.
        # Search options
        super(RankPrune, self).__init__(leaf_eval)

        self.evaluation_function = None
        self.leaf_eval = leaf_eval
        self.branch_eval = branch_eval if branch_eval is not None else leaf_eval
        self.prune_perc = prune_perc
        self.time_limit = time_limit
        self.buff_time = buff_time
        self.limit_depth = limit_depth
        self.max_depth = max_depth
        self.leaf_mode = False

        # Generate an estimate of calling the leaf_eval function.
        start = time.time()
        num_estimate = 10
        for _ in range(num_estimate):
            self.leaf_eval(chess.Board().fen())
        self.leaf_estimate = (time.time() - start) / num_estimate

    def condition(self, depth):
        # TODO: Perhaps this should return the evaluatoin function? I don't think it should set it, its confusing.
        # TODO (continued): It's also confusing because this doesn't do what the abstract method implies it does.
        # TODO (continued): Here what it really does is set the evaluation function, not check if it SHOULD be evaluated.
        # TODO: Suggestion: split 'condition' into 'condition' and 'evaluation_selector'

        if self.leaf_mode or (self.limit_depth and depth == self.max_depth):
            self.evaluation_function = self.leaf_eval
        else:
            self.evaluation_function = self.branch_eval
        return True

    def run(self, board, time_limit=None, clear_cache=False):
        """
            BFS-based search function which does breadth first evaluation of boards. Every child is evaluated using the
            'inner_eval' function. The children are then ranked and 'prune_perc' are pruned. The non-pruned children
            are iteratively expanded upon and have their score updated based on this expansion. When time runs out
            leaf nodes are evaluated using 'leaf_eval' function.
            Inputs:
                root_board [chess.Board]:
                    current state of board
                time_limit [Float] (Optional)
                    Maximum search time. If not specified, uses the class default.
                limit_depth [Boolean] (Optional)
                    If True then rank_prune depth is limited by self.max_depth. If False there is not depth limit.
            Outputs:
                best_score [float]:
                    Score achieved by best move
                best_move [chess.Move]:
                    Best move to play
                best_leaf [String]
                    FEN of the board of the leaf node which yielded the highest value.
        """
        if time_limit is None:
            time_limit = self.time_limit

        if time_limit <= self.buff_time:
            raise ValueError("Rank-Prune Error: Time limit (%d) <= buffer time (%d). "
                             "Please increase time limit or reduce buffer time." % (time_limit, self.buff_time))

        if clear_cache:
            self.cache = {}

        # Start timing
        start_time = time.time()

        score, _, fen = self.check_conditions(board, 0)

        root = SearchNode(fen, 0, score)  # Note: store unflipped fen

        # Evaluation Queue. Holds SearchNode
        queue = Queue.Queue()
        queue.put(root)

        self.leaf_mode = False
        while not queue.empty():
            curr_node = queue.get()
            board = chess.Board(curr_node.fen)

            # If depth is limited and maximum depth is reached, skip expansion
            if self.limit_depth and curr_node.depth == self.max_depth:
                continue

            if self.leaf_mode:
                # Re-evaluate node with leaf_eval
                curr_node.value = self.check_conditions(board, curr_node.depth)[0]
            else:
                # Evaluate all children
                for i, move in enumerate(board.legal_moves):

                    # Check if time-out
                    time_left = time_limit - self.buff_time - (time.time() - start_time)
                    to_eval = queue.qsize() + len(board.legal_moves) - i  # Number of nodes to leaf_eval if stopped now
                    if not self.leaf_mode and time_left <= to_eval * self.leaf_estimate:
                        # clear queue and break
                        # print "Running out of time on depth %d. Queue size %d " % (curr_node.depth + 1, queue.qsize())
                        self.leaf_mode = True

                    # play move
                    board.push(move)
                    score, _, fen = self.check_conditions(board, curr_node.depth)
                    # Note: Store unflipped fen
                    curr_node.add_child(move, SearchNode(fen, curr_node.depth + 1, score))

                    # Undo move
                    board.pop()

                # Expand if didn't time out
                if not self.leaf_mode:
                    # Prune children if necessary
                    if len(curr_node.children) > 1:
                        # TODO: Maybe this is better done in place
                        top_ranked = k_top(curr_node.get_child_nodes(),
                                           int(math.ceil(len(curr_node.children) * (1 - self.prune_perc) + 1)),
                                           key=lambda x: x.value)
                    else:
                        top_ranked = curr_node.get_child_nodes()
                    # Queue non-pruned children
                    for child in top_ranked:
                        queue.put(child)

        # Minimax on game tree
        return minimaxtree(root)

    def __str__(self):
        return "RankPrune"


class SearchNode:
    def __init__(self, fen, depth, value):
        """
        Generic node used for searching in 'rank_prune'.
        Input:
            fen [String]
                FEN of the board.
            depth [Int]
                Depth at which the node occurs.
            value [Float]
                Value of the node, i.e. P(current player wins).
        Non-Input Class Attributes
            children [Dict of SearchNode's]
                Children of the current SearchNode, key is Move which yields that SearchNode.

        """
        self.fen = fen
        self.depth = depth
        self.value = value
        self.children = {}

    def add_child(self, move, child):
        assert (isinstance(child, SearchNode))

        self.children[move] = child

    def get_child_nodes(self):
        """
        Returns a list of child nodes.
        """
        return self.children.values()

    def __str__(self):
        return "Node{%s, %d, %f, %d children}" % (self.fen, self.depth, self.value, len(self.children))


def k_top(arr, k, key=None):
    """
    Selects the k-highest valued elements from the input list. i.e. k = 1 would yield the maximum element.
    Uses quickselect.
    Input:
        arr [List]
            List of items from which the k-highest valued items should be selected.
        k [Int]
            The number of highest valued items to be returned.
        key [Function] (Optional)
            Used to convert items in arr to values. Useful when dealing with objects. Default is identity function.
    Output:
        output [List]
            k-highest valued items.
    """
    if k <= 0:
        raise ValueError("K-Top error: %d is an invalid value for k, k must be > 0." % k)

    if len(arr) < k:
        return arr

    # Identity
    if key is None:
        key = lambda x: x

    # quickselect -> also partitions
    quickselect(arr, 0, len(arr) - 1, len(arr) - k, key=key)

    return arr[-k:]


def quickselect(arr, left, right, k, key=None):
    """
    Return the array with the k-th smallest valued element at the k-th index.
    Note that this requires left <= k <= right.
    Based on: https://en.wikipedia.org/wiki/Quickselect
    Input:
        arr [List]
            List of items from which the k-highest valued items should be selected.
        left [Int]
            Starting index of the array to be considered.
        right [Int]
            Ending index of the array to be considered. (Inclusive)
        k [Int]
            The number of highest valued items to be returned.
        key [Function] (Optional)
            Used to convert items in arr to values. Useful when dealing with objects. Default is identity function.
    Output:
        output [List]
            k-highest valued items.
    """

    assert (left <= k <= right)

    if key is None:
        key = lambda x: x

    while True:
        # Base case -> If one element then return that element
        if left == right:
            return arr[right]
        pivot_idx = random.randint(left, right)
        pivot_idx = partition(arr, left, right, pivot_idx, key=key)
        if k == pivot_idx:
            return arr[k]
        elif k < pivot_idx:
            right = pivot_idx - 1
        else:
            left = pivot_idx + 1


def partition(arr, left, right, pivot_idx, key=None):
    # TODO: Test
    """
    Partitions array inplace into two parts, those smaller than pivot, and those larger.
    Based on partition pseudo-code from: https://en.wikipedia.org/wiki/Quickselect
    Input:
        arr [List]
            List of items to be partitioned.
        left [Int]
            Starting index of the array to be partitioned.
        right [Int]
            Ending index of the array to be partitioned. (Inclusive)
        pivot_idx [Int]
            Index of the pivot.
        key [Function] (Optional)
            Used to convert items in arr to values. Useful when dealing with objects. Default is identity function.
    Output:
        store_idx [Int]
            Pivot location.
    """

    if key is None:
        key = lambda x: x

    pivot_val = key(arr[pivot_idx])

    # move pivot to end
    arr[pivot_idx], arr[right] = arr[right], arr[pivot_idx]

    store_idx = left
    for i in range(left, right):
        if key(arr[i]) < pivot_val:
            arr[store_idx], arr[i] = arr[i], arr[store_idx]
            store_idx += 1
    # Move pivot into right place
    arr[right], arr[store_idx] = arr[store_idx], arr[right]

    # Return pivot location
    return store_idx


def minimaxtree(root, a=1.0, forbidden_fens=None):
    """
        Recursive function to find for best move in a game tree using minimax with alpha-beta pruning.
        Assumes that the layer above the leaves are trying to minimize the positive value,
        which is the same as maximizing the reciprocal.
        Inputs:
            board [chess.Board]:
                current state of board
            a [float]:
                lower bound of layer above, upper bound of current layer (because of alternating signs)
            forbidden_fens [List of Strings] (Optional)
                List of FENs. An error is thrown if a node is reached which has one of these FENs. Used for testing.
                Default is to have no fens forbidden.
        Outputs:
            best_score [float]:
                Score achieved by best move
            best_move [chess.Move]:
                Best move to play
            best_leaf [String]
                FEN of the board of the leaf node which yielded the highest value.
    """
    assert (isinstance(root, SearchNode))

    best_score = 0.0
    best_move = None
    best_leaf = None

    # check if forbidden fen
    if forbidden_fens and root.fen in forbidden_fens:
        raise RuntimeError("Minimaxtree Error: Forbidden FEN %s reached!" % root.fen)

    # Check if leaf
    if not root.children:
        return root.value, None, root.fen

    else:
        for move, child in root.children.iteritems():
            score, next_move, leaf_board = minimaxtree(child, 1 - best_score, forbidden_fens=forbidden_fens)
            # Take reciprocal of score since alternating levels
            score = 1 - score
            if score >= best_score:
                best_score, best_move, best_leaf = score, move, leaf_board

            # best_score is my current lower bound
            # a is the upper bound of what's useful to search
            # if my lower bound breaks the boundaries of what's worth to search
            # stop searching here
            if best_score >= a:
                break

    # print "D%d: best: %.1f, %s" % (depth, best_score, best_move)
    return best_score, best_move, best_leaf
