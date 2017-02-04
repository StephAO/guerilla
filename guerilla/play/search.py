import time
import Queue
import chess
import random

import guerilla.data_handler as dh
import guerilla.play.node as mc_node


class Search:
    """
    Implements game tree search.
    """

    def __init__(self, leaf_eval, inner_eval=None, max_depth=2, search_mode="complementmax"):
        # Evaluation function must yield a score between 0 and 1.
        # Search options
        self.search_opts = {"complementmax": self.complementmax,
                            "rank_prune": self.rank_prune,
                            "montecarlo": self.monte_carlo}

        if search_mode not in self.search_opts:
            raise NotImplementedError("Invalid Search option!")
        self.search_mode = search_mode

        self.leaf_eval = leaf_eval
        self.max_depth = max_depth
        self.win_value = 1
        self.lose_value = 0
        self.draw_value = 0.5
        self.reci_prune = True

        # Rank Prune Parameters
        self.inner_eval = inner_eval if inner_eval is not None else leaf_eval
        self.perc_prune = 0.5

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

    def complementmax(self, board, depth=0, a=1.0, cache=None):
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
                cache [Dictionary]
                    Cache of previous results.
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

        # Create cache if necessary
        if cache is None:
            cache = {}

        # Check if draw
        if board.is_checkmate():
            return self.lose_value, None, board.fen()
        elif board.can_claim_draw() or board.is_stalemate():
            return self.draw_value, None, board.fen()
        elif depth == self.max_depth:
            fen = leaf_board = board.fen()
            if dh.black_is_next(fen):
                fen = dh.flip_board(fen)

            # Check for cache hit
            if fen not in cache:
                cache[fen] = self.leaf_eval(fen)
            return cache[fen], None, leaf_board

        else:
            for move in board.legal_moves:
                # print "D%d: %s" % (depth, move)
                # recursive call
                board.push(move)
                # print move
                score, next_move, leaf_board = self.complementmax(board, depth + 1, 1 - best_score, cache=cache)
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

    # ---------- RANK PRUNE METHODS

    def rank_prune(self, root_board, time_limit=10, buff_time=0.5):
        """
            BFS-based search function which does breadth first evaluation of boards. Every child is evaluated using the
            'inner_eval' function. The children are then ranked and 'perc_prune' are pruned. The non-pruned children
            are iteratively expanded upon and have their score updated based on this expansion. When time runs out
            leaf nodes are evaluated using 'leaf_eval' function.
            Inputs:
                root_board [chess.Board]:
                    current state of board
                time_limit [Float] (Optional)
                    Maximum search time.
                buff_time [Float] (Optional)
                    Buffer time allocated to finding the global maximum once the search tree has been built.
            Outputs:
                best_score [float]:
                    Score achieved by best move
                best_move [chess.Move]:
                    Best move to play
                best_leaf [String]
                    FEN of the board of the leaf node which yielded the highest value.
        """
        #TODO: Test

        # Start timing
        start_time = time.time()

        root = SearchNode(root_board.fen(), 0, self.inner_eval(root_board.fen()))

        # Evaluation Queue. Holds SearchNode
        queue = Queue.Queue()
        queue.put(root)

        # Cache: Key is FEN, Value is Score
        cache = {}

        curr_depth = None
        expand_start = None
        expand_time = 1.0 # Conservative initial estimate -> updated later
        first_expand = True
        is_leaf = False
        while not queue.empty():
            curr_node = queue.get()
            board = chess.Board(curr_node.fen)

            # Check if reached a new depth, update if necessary
            if curr_depth is None or curr_depth != curr_node.depth:
                curr_depth = curr_node.depth

                # Check if have time to finish, if not then switch to leaf_eval
                time_left = time_limit - (time.time() - start_time) - buff_time
                if time_left < expand_time*queue.qsize():
                    print "Running out of time on depth %d" % curr_depth
                    is_leaf = True

            # If first layer then time it
            if first_expand:
                expand_start = time.time()

            # Evaluate all children
            for move in board.legal_moves:
                # play move
                board.push(move)

                # Flip board if necessary
                fen = board.fen()
                if dh.black_is_next(fen):
                    fen = dh.flip_board(fen)

                # Evaluate board using evaluation function or use cache if possible, but always replace with leaf eval
                if fen not in cache or is_leaf:
                    # Check if draw or checkmate, otherwise evaluate
                    if board.is_checkmate():
                        cache[fen] = self.lose_value
                    elif board.can_claim_draw() or board.is_stalemate():
                        cache[fen] = self.draw_value
                    else:
                        # Evaluate, flip if necessary
                        cache[fen] = self.leaf_eval(fen) if is_leaf else self.inner_eval(fen)

                # Add as child
                curr_node.add_child(move, SearchNode(fen, curr_node.depth + 1, cache[fen]))

                # Check if time-out
                if time.time() - start_time + buff_time > time_limit:
                    # clear queue and break
                    queue = Queue.Queue()
                    break

                # Undo move
                board.pop()

            # Save timing of first layer
            if first_expand:
                expand_time = time.time() - expand_start
                first_expand = False

            # Expand if not on last layer
            if not is_leaf:
                # Prune children
                # TODO: Maybe this is better done in place
                top_ranked = k_top(curr_node.get_child_nodes(), int(len(curr_node.children)*self.perc_prune),
                                                  key=lambda x:x.value)

                # Queue non-pruned children
                for child in top_ranked:
                    queue.put(child)


        # Minimax on game tree
        return minimaxtree(root)

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
        assert(isinstance(child, SearchNode))

        self.children[move] = child

    def get_child_nodes(self):
        """
        Returns a list of child nodes.
        """
        return self.children.values()

    def __str__(self):
        return "Node{%s, %d, %f, %d children}" % (self.fen, self.depth, self.value, len(self.children))

def k_top(arr, k, key=None):
    # TODO: Test
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

    # Identity
    if key is None:
        key = lambda x: x

    # quickselect -> also partitions
    quickselect(arr, 0, len(arr) - 1, len(arr) - k, key = key)

    return arr[k:]

def quickselect(arr, left, right, k, key=None):
    #TODO: Test
    """
    Return the k-th smallest valued element from the input list.
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

def minimaxtree(root, a=1.0):
    """
        Recursive function to find for best move in a game tree using minimax with alpha-beta pruning.
        Assumes that the layer above the leaves are trying to minimize the positive value,
        which is the same as maximizing the reciprocal.
        Inputs:
            board [chess.Board]:
                current state of board
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
    assert(isinstance(root,SearchNode))

    best_score = 0.0
    best_move = None
    best_leaf = None

    # Check if leaf
    if root.children == []:
        return root.value, None, root.fen

    else:
        for move, child in root.children.iteritems():
            score, next_move, leaf_board = minimaxtree(child, 1 - best_score)
            # Take reciprocal of score since alternating levels
            score = 1 - score
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

    # print "D%d: best: %.1f, %s" % (depth, best_score, best_move)
    return best_score, best_move, best_leaf