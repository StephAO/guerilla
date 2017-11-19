from abc import ABCMeta, abstractmethod
import chess
import math
import time
import numpy as np

from search_helpers import quickselect, material_balance
import guerilla.data_handler as dh

class Search:
    __metaclass__ = ABCMeta

    def __init__(self, evaluation_function):
        """
        Initializes the Player abstract class.
        Input:
            evaluation_function [function]:
                function to evaluate leaf nodes (usually the neural net)
        """
        self.evaluation_function = evaluation_function
        self.win_value = dh.WIN_VALUE
        self.lose_value = dh.LOSE_VALUE
        self.draw_value = dh.DRAW_VALUE
        self.num_evals = 0
        self.num_visits = []  # Index: depth, Value: Number of vists at that depth
        self.depth_reached = 0
        self.cache_hits = 0
        self.cache_miss = 0
        # cache: Key is FEN of board (where white plays next), Value is Score
        self.cache = {}
        self.eval_time = 0  # Time spent evaluating boards

    @abstractmethod
    def __str__(self):
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

    # TODO: This should be part of transposition table
    def conditional_eval(self, board, depth, **kwargs):
        """ 
        Check current state of board, and decided whether or not to evaluate
        the given board.
        Inputs:
            board[chess.Board]:
                current board
            depth[int]:
                current depth
            kwargs[dict]:
                necessary data for evaluation condition and selection
        Outputs:
            score [float]:
                Score achieved by the board. P(win of next player to play)
            best_move [chess.Move]:
                Best move to play
            best_leaf [String]
                FEN of the board of the leaf node which yielded the highest value.
        """
        try:
            self.num_visits[depth] += 1
        except IndexError:
            self.num_visits.append(1)

        # check if new depth reached
        if depth > self.depth_reached:
            self.depth_reached = depth

        # add for inputs
        kwargs['depth'] = depth

        fen = unflipped_fen = board.fen()
        fen = dh.flip_to_white(fen)
        # Check if draw
        if board.is_checkmate():
            return self.lose_value, None, unflipped_fen
        elif board.can_claim_draw() or board.is_stalemate():
            return self.draw_value, None, unflipped_fen
        else:
            # Check for self.cache hit
            # Use cache
            key = dh.strip_fen(fen)
            self.num_evals += 1
            if key not in self.cache:
                self.cache_miss += 1
                start = time.time()
                self.cache[key] = self.evaluation_function(fen)
                self.eval_time += time.time() - start
            else:
                self.cache_hits += 1
            score = self.cache[key]

            return score, None, unflipped_fen

    def clear_cache(self):
        # Clears the cache
        self.cache = {}


class IterativeDeepening(Search):
    """
    Searches game tree in an Iterative Deepening Depth search.
    At each depth optionally prune from remaining possibilities
    Implements alpha_beta pruning by default
    """
    def __init__(self, evaluation_function, time_limit=10,
                 max_depth=None, h_prune=False, prune_perc=0.0,
                 ab_prune=True, verbose=True, use_partial_search=False):

        """
            Constructor
            Inputs:
                evaluation_function[function]:
                    function used to evaluate leaf nodes
                time_limit[float]:
                    time limit per move
                max_depth[int]:
                    If not None, limit depth to max_depth
                h_prune[bool]:
                    Heuristic_prune. If true, prune between between depth-limited-searches
                prune_perc[float range([0,1])]:
                    Percent of nodes to prune for heuristic prune
                ab_prune[bool]:
                    Alpha-beta pruning. Same results on or off, only set to off for td training
                use_partial_search [Bool]:
                    Whether or not to use partial search results (i.e. when a timeout occurs during DFS).
        """
        super(IterativeDeepening, self).__init__(evaluation_function)

        self.time_limit = time_limit
        self.buff_time = time_limit * 0.02
        self.depth_limit = 1
        self.max_depth = max_depth
        self.h_prune = h_prune
        self.prune_perc = prune_perc
        self.ab_prune = ab_prune
        self.root = None  # holds root node
        self.order_fn_fast = material_balance  # The move ordering function to use pre-Guerilla evaluation
        self.order_moves = True
        self.is_partial_search = False  # Marks if the last DFS call was a partial search
        self.use_partial_search = use_partial_search

        # Propagation functions, for normal minimax use 0
        self.use_prop = True
        self.up_prop_decay = 0  # decay value propagating upwards (0 for normal minimax -> pass up child value w/o decay)
        self.down_prop_decay = 1.0  # decay value propagating downwards (1 for normal minimax -> only use child value, decay parent value entirely)

        # Holds the Killer Moves by depth. Each Entry is (set of moves, sorted array of (score, moves)).
        self.killer_table = [{'moves': set(), 'scores': list()}]
        self.num_killer = 2  # Number of killer moves store for each depth

    def __str__(self):
        return "IterativeDeepening"

    def _up_prop(self, child_value, curr_value):
        if self.use_prop:
            return curr_value * (self.up_prop_decay) + child_value * (1 - self.up_prop_decay)
        return child_value

    def _down_prop(self, parent_value, curr_value):
        if self.use_prop:
            return curr_value * (self.down_prop_decay) + parent_value * (1 - self.down_prop_decay)
        return curr_value

    def DLS(self, node, parent=None, a=float("inf")):
        """
            Recursive depth limited search with alpha_beta pruning.
            Assumes that the layer above the leaves are trying to minimize the positive value,
            which is the same as maximizing the reciprocal.
            Inputs:
                node [SearchNode]:
                    Node to expand
                parent [SearchNode]
                    Parent Node. If None, then no parent.
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
        best_move = None
        best_score = -float("inf")
        leaf_fen = None
        board = chess.Board(node.fen)

        self.time_left = (time.time() - self.start_time) <= self.time_limit - self.buff_time
        if node.depth >= self.depth_limit or not node.expand or not self.time_left:
            if not self.time_left:
                self.is_partial_search = True
            return self._down_prop(node.value if parent is None else parent.value, node.value), None, node.fen
        else:
            # Check if a new killer table entry should be created
            if node.depth >= len(self.killer_table):
                self.killer_table.append({'moves': set(), 'scores': list()})

            moves = list(board.legal_moves)
            if self.order_moves:
                killers = []
                fast_order = []
                node_order = []
                for move in moves:

                    # Child has not previously been evaluated, use fast scoring & ordering

                    # TODO: Transposition has an effect here

                    if move not in node.children:
                        # Get scores
                        board.push(move)
                        value = self.order_fn_fast(board.fen())
                        fast = True
                        board.pop()
                    else:
                        # Child has previously been evaluated, use node evaluation for ordering
                        value = node.children[move].value
                        fast = False

                    # Check which ordering it should go into
                    move_inf = (move, value, fast)
                    if self.is_killer(move, node.depth):
                        killers.append(move_inf)
                    elif move not in node.children:
                        fast_order.append(move_inf)
                    else:
                        node_order.append(move_inf)

                # Order in ascending order (want to check boards which are BAD for opponent first)
                killers.sort(key=lambda x: x[1])
                node_order.sort(key=lambda x: x[1])
                fast_order.sort(key=lambda x: x[1])

                # Favor killer moves the most
                # Always favor node ordering, mark whether node needs to be evaluated
                moves = [(x[0], x[2]) for x in killers + node_order + fast_order]

            for move, new_move in moves:
                # Generate child if necessary
                if new_move:
                    node.gen_child(move, eval_fn=self.conditional_eval)

                # print move
                # TODO: [TT] DLS should not becalled if already in transposition table -> Maybe replace DLS call w/ wrapper that handles Transposition
                score, next_move, lf = self.DLS(node.children[move], parent=node, a=-best_score)
                score = -score
                # Best child is the one one which has the lowest value
                if best_move is None or score > best_score:
                    best_score = score
                    best_move = move
                    leaf_fen = lf

                if self.ab_prune and best_score >= a:
                    self.update_killer(move, best_score, node.depth)
                    break

            node.value = self._up_prop(child_value=best_score, curr_value=node.value)

        return node.value, best_move, leaf_fen

    def update_killer(self, killer_move, score, depth):
        """
        Updates the killer move table.
        Input:
            killer_move [Chess.Move]
                The move which caused the A-B pruning to trigger.
            score [Float]
                The score yielded by the killer move.
            depth [Int]
                The depth FROM which the move was played.
        """

        k_tab = self.killer_table[depth]

        # Skip if already in killer table
        if killer_move in k_tab['moves']:
            return

        # Check if table is full
        if len(k_tab['moves']) < self.num_killer:
            # Not Full
            self._add_killer_move(depth, score, killer_move)
        else:
            # Full
            # Check if move is better than worst current move
            if score > k_tab['scores'][0]:
                # Remove Worst
                _, worst_move = k_tab['scores'].pop(0)
                k_tab['moves'].remove(worst_move)

                # Add Item
                self._add_killer_move(depth, score, killer_move)

    def _add_killer_move(self, depth, score, killer_move):
        """
        Adds killer move to the table.
        """
        # Update moves
        self.killer_table[depth]['moves'].add(str(killer_move))

        # Update scores
        self.killer_table[depth]['scores'].append((score, str(killer_move)))
        self.killer_table[depth]['scores'].sort(key=lambda x: x[0])  # Sorting each time is OK since length is small.

    def is_killer(self, move, depth):
        """
        Checks if the current move is a killer move.
        Input:
            move [chess.Move]
                Move to check.
        Output:
            output [Boolean]
                True if it is a kill move, False if not.
        """
        return str(move) in self.killer_table[depth]['moves']

    def prune(self, node):
        """
            Recursive pruning of nodes
        """
        if not node.expand or not node.children:
            return

        children = list(node.get_child_nodes())
        # k = number of nodes that I keep
        k = max(min(len(children), 2),
                int(math.ceil(len(children) * (1 - self.prune_perc))))
        quickselect(children, 0, len(children) - 1, k - 1, key=lambda x: x.value)

        for child in children[:k]:
            self.prune(child)
            child.expand = True
        for child in children[k:]:
            child.expand = False

    def run(self, board, time_limit=None, clear_cache=False):
        """
            For the duration of the time limit and depth limit:
                1. Depth Limited Search
                2. If enabled: Prune nodes
                3. Increase max depth
            Inputs:
                board[chess.Board]:
                    Chess board to search the best move for
                time_limit[float]:
                    time limit for search. If None, defaults to time_limit set in init
                clear_cache[bool]:
                    clear evaluation cache. Needed for trainings
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

        if clear_cache:
            self.clear_cache()
        self.num_evals = 0
        self.eval_time = 0
        self.num_visits = []

        # Start timing
        if time_limit is not None:
            self.time_limit = time_limit
        self.start_time = time.time()
        self.time_left = True

        score, _, fen = self.conditional_eval(board, depth=0)

        self.root = SearchNode(fen, 0, score)  # Note: store unflipped fen

        self.depth_limit = 1
        score = best_move = leaf_board = None
        while self.time_left and (self.max_depth is None or self.depth_limit <= self.max_depth):

            # Run search
            new_results = self.DLS(self.root)
            if not self.is_partial_search or (self.is_partial_search and self.use_partial_search):
                score, best_move, leaf_board = new_results
            self.is_partial_search = False

            # Prune if necessary
            if self.h_prune:
                self.prune(self.root)

            # Increase depth
            self.depth_limit += 1

        return score, best_move, leaf_board


class SearchNode:
    def __init__(self, fen, depth):
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
        self.value = None  # TODO: Value comes from transposition table
        self.children = {}
        self.expand = True

    def add_child(self, move, child):
        assert (isinstance(child, SearchNode))

        self.children[move] = child

    def gen_child(self, move, eval_fn):
        """
        Generates a child for the search node based on the input move.
        Generation involves adding and evaluating the child.
        Throws an error if move is not legal or if input move has already been used for child generation.
        Input:
            move [chess.Move]
                Move used for child generation. Must be a legal move which has not already been generated.
            eval_fn [Function]
                Function used to evaluate the child.
        """
        # TODO: Populate with transposition table results
        board = chess.Board(self.fen)
        if not board.is_legal(move):
            raise ValueError("Invalid move for child generation! %s is not a legal move from %s.", str(move), self.fen)
        if move in self.children:
            raise ValueError("Invalid move for child generation! Child Node %s already exists", str(move))
        board.push(move)
        score, _, fen = eval_fn(board, depth=self.depth + 1)
        self.add_child(move, SearchNode(fen, self.depth + 1, score))
        board.pop()


    def get_child_nodes(self):
        """
        Returns a list of child nodes.
        """
        return self.children.values()

    def __str__(self):
        return "Node{%s, %d, %f, %d children}" % (self.fen, self.depth, self.value, len(self.children))


class Minimax(IterativeDeepening):
    """
        Uses a recursive function to perform a simple minimax with
        alpha-beta pruning.
    """

    def __init__(self, leaf_eval, max_depth=2, ab_prune=True):
        """
            Constructor
            Inputs:
                leaf_eval[function]:
                    function used to evaluate leaf nodes
                max_depth[int]:
                    depth to go to
                ab_prune[bool]:
                    Alpha-beta pruning. Same results on or off, only set to off for td training
        """
        super(Minimax, self).__init__(leaf_eval, time_limit=np.inf, max_depth=max_depth,
                                      prune_perc=0.0, ab_prune=ab_prune, use_partial_search=False)

        # Turn off propagation
        self.use_prop = False

        # No buffer time needed
        self.buff_time = 0

    def __str__(self):
        return "Minimax"

    def run(self, board, time_limit=None, clear_cache=False):
        """ Reset some variables, call recursive function """
        if clear_cache:
            self.clear_cache()
        self.num_evals = 0
        self.eval_time = 0
        self.num_visits = []
        self.time_left = True

        # Run to depth limit
        self.start_time = 0
        self.root = SearchNode(board.fen(), 0, None)  # Note: store unflipped fe
        return self.DLS(self.root)
