from abc import ABCMeta, abstractmethod
import chess
import math
import time
import numpy as np
from collections import namedtuple

from search_helpers import quickselect, material_balance
import guerilla.data_handler as dh

# Note:
#   Moves are stored as UCI strings
#   Leaf FEN is stripped
Transposition = namedtuple("Transposition", "best_move value leaf_fen node_type")

PV_NODE = 0
CUT_NODE = 1
ALL_NODE = 2
LEAF_NODE = 3  # Additional type, we use it to demark nodes which were leafs to reduce the # of evaluation


class TranspositionTable:
    def __init__(self, exact_depth=True):
        self.table = {}  # Main transposition table {FEN: Transposition Entry}
        self.exact_depth = exact_depth  # If we require an exact depth match

        # TODO
        self.cache_hits = 0
        self.cache_miss = 0

    def fetch(self, fen, requested_depth):
        # Returns {Best Move, Value, Type, Depth}

        # Check for entry in table
        white_fen = dh.flip_to_white(fen)
        # print "Fetching ORIGINAL {} WHITE {}".format(fen, white_fen)
        entry = self._get_entry(white_fen)
        if not entry or entry.deepest < requested_depth:
            self.cache_miss += 1
            return

        # If not using exact depth, then get deepest depth
        if not self.exact_depth:
            requested_depth = entry.deepest

        # Fetch depth result from table
        if requested_depth in entry.value_dict:
            self.cache_hits += 1
            transpo = entry.value_dict[requested_depth]

            # If black is next then flip move AND leaf fen
            if dh.black_is_next(fen):
                transpo = self._flip_transposition(transpo)

            return transpo
        else:
            self.cache_miss += 1

    def update(self, fen, search_depth, best_move, score, leaf_fen, node_type):

        # Flip to white
        transpo = Transposition(best_move, score, leaf_fen, node_type)

        # Flip transposition if necessary
        if dh.black_is_next(fen):
            transpo = self._flip_transposition(transpo)

        entry = self._get_entry(dh.flip_to_white(fen), create=True)
        entry.add_depth(search_depth, transpo)

    def exists(self, fen):
        assert (dh.white_is_next(fen))
        return dh.strip_fen(fen) in self.table

    def _get_entry(self, fen, create=False):
        assert (dh.white_is_next(fen))
        key = dh.strip_fen(fen)
        if key not in self.table and create:
            self.table[key] = TranpositionEntry()
        return self.table.get(key)

    def _flip_transposition(self, entry):
        best_move = dh.flip_move(entry.best_move) if entry.best_move is not None else entry.best_move
        leaf_fen = dh.flip_board(entry.leaf_fen)
        return Transposition(best_move, entry.value, leaf_fen, entry.node_type)

    def clear(self):
        # Clears the transposition table
        self.table.clear()
        self.cache_hits = self.cache_miss = 0

    def best_estimate(self, fen):
        """
        Returns the current best estimate for the input FEN (deepest score).
        :param fen: [String] FEN.
        :return: [Transposition] If FEN exists in cache else [None]
        """

        # Get entry
        white_fen = dh.flip_to_white(fen)
        entry = self._get_entry(white_fen)

        if entry is None:
            return entry

        # Get deepest transposition
        transpo = entry.value_dict[entry.deepest]

        # Flip if necessary
        if dh.black_is_next(fen):
            transpo = self._flip_transposition(transpo)

        return transpo


class TranpositionEntry:
    def __init__(self):
        self.value_dict = {}  # {Depth: {Best Move (UCI), Value, Leaf FEN, Node Type,}}
        self.deepest = None

    def add_depth(self, depth, transposition):
        # Update value dict and deepest
        if depth in self.value_dict:
            raise ValueError("Should never try to update an existing depth!")
        self.value_dict[depth] = transposition

        self.deepest = max(depth, self.deepest)


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
        # Transposition table
        self.tt = TranspositionTable(exact_depth=True)  # TODO: We can change this

    def evaluate(self, fen):
        # Returns the value of the input FEN from the perspective of the next player to play
        # Depth is used to update information

        self.num_evals += 1

        # Evaluate
        board = chess.Board(fen)
        if board.is_checkmate():
            return self.lose_value
        elif board.can_claim_draw() or board.is_stalemate():
            return self.draw_value
        else:
            return self.evaluation_function(dh.flip_to_white(fen))

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
                best_value [float]:
                    value achieved by best move
                best_leaf [String]
                    FEN of the board of the leaf node which yielded the highest value.
        """
        raise NotImplementedError("You should never see this")

    def reset(self):
        # Clears the transposition table
        self.tt.clear()

        # Reset some logging variables
        self.num_evals = 0
        self.num_visits = []
        self.depth_reached = 0


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
        self.depth_limit = 1  # Depth limit for DFS
        self.max_depth = max_depth
        self.order_moves = True  # Whether moves should be ordered
        self.h_prune = h_prune
        self.prune_perc = prune_perc
        self.ab_prune = ab_prune
        self.root = None  # holds root node
        self.is_partial_search = False  # Marks if the last DFS call was a partial search
        self.use_partial_search = use_partial_search

        # Holds the Killer Moves by depth. Each Entry is (set of moves, sorted array of (value, moves)).
        self.killer_table = [{'moves': set(), 'values': list()}]
        self.num_killer = 2  # Number of killer moves store for each depth

        # Move value for ordering when board not found in transposition table
        self.order_fn_fast = material_balance

    def __str__(self):
        return "IterativeDeepening"

    def _uci_to_move(self, uci):
        return chess.Move.from_uci(uci) if uci is not None else uci

    def DLS(self, node, alpha, beta):
        """
        Recursive depth limited negamax search with alpha_beta pruning.
        Source: https://en.wikipedia.org/wiki/Negamax#Negamax_with_alpha_beta_pruning_and_transposition_tables
        :param node: [SearchNode] Roote node for search.
        :param alpha: [Float] Lower bound.
        :param beta: [Float] Upper bound.
        :return:
            best_score [float]:
                Score achieved by best move
            best_move [chess.Move]:
                Best move to play
            best_leaf [String]
                FEN of the board of the leaf node which yielded the highest value.
        """

        # Track some info
        if not node.visited:
            # We've never seen this node before -> Track some info
            try:
                self.num_visits[node.depth] += 1
            except IndexError:
                self.num_visits.append(1)

            self.depth_reached = max(self.depth_reached, node.depth)

            # Check if a new killer table entry should be created
            if node.depth >= len(self.killer_table):
                self.killer_table.append({'moves': set(), 'values': list()})

            node.visited = True

        alpha_original = alpha

        # Check transposition table
        result = self.tt.fetch(node.fen, requested_depth=self.depth_limit - node.depth)
        if result:
            if result.node_type == PV_NODE:
                return result.value, result.best_move, result.leaf_fen
            elif result.node_type == CUT_NODE:
                # lower bound
                alpha = max(alpha, result.value)
            elif result.node_type == ALL_NODE:
                # upper bound
                beta = min(beta, result.value)

            if alpha >= beta:
                return result.value, self._uci_to_move(result.best_move), result.leaf_fen

        # Check children
        if not node.children:
            node.gen_children()

        # Check if limit reached
        self.time_left = (time.time() - self.start_time) <= self.time_limit - self.buff_time
        if node.depth == self.depth_limit or not node.expand or not self.time_left or not node.children:
            # Evaluate IF: depth limit reached, pruned, no time left OR no children
            if not self.time_left:
                self.is_partial_search = True

            # Check if we have previously evaluated this node as a leaf node
            if result and result.node_type == LEAF_NODE:
                return result.value, None, node.fen

            # Otherwise evaluate node
            node.value = self.evaluate(node.fen)

            # Update transposition table
            self.tt.update(node.fen, self.depth_limit - node.depth, None, node.value, node.fen, LEAF_NODE)

            return node.value, None, node.fen

        # Get Ordered children
        moves = self.get_ordered_moves(node) if self.order_moves else node.child_moves

        # Find best move (recursive)
        best_value = float("-inf")
        best_move = None
        leaf_fen = None

        for move in moves:
            # Get best score for opposing player and flip it to your perspective
            value, _, lf = self.DLS(node.children[move], alpha=-beta, beta=-alpha)
            value = -value

            if value > best_value:
                best_move = move
                best_value = value
                leaf_fen = lf

            # Check for pruning
            alpha = max(alpha, value)
            if alpha >= beta:
                self.update_killer(move, value, node.depth)
                break

        # Update transposition table
        if best_value <= alpha_original:
            # ALL NODES searched, no good moves found -> value is an upper bound
            node_type = ALL_NODE
        elif best_value >= beta:
            # CUT NODE, pruning occurred -> value is a lower bound
            node_type = CUT_NODE
        else:
            # PV NODE Otherwise its potentially part of the the principal variation
            node_type = PV_NODE

        # Update transposition table
        self.tt.update(node.fen, self.depth_limit - node.depth, best_move, best_value, leaf_fen, node_type)

        # Return result of search
        return best_value, self._uci_to_move(best_move), leaf_fen

    def get_ordered_moves(self, node):
        """
        Orders the child moves of the node.
        Ordering is done as follows:
            (*) Leaf nodes if at max depth
            (2) PV Nodes
            (3) Cut Nodes
            (*) Leaf Nodes if NOT at max depth
            (5) Killer moves
            (6) All Nodes (note: "All Nodes" are specific type of node)
            (7) Other nodes
        :param node: [SearchNode] Node who's child moves we need to order.
        :return: [List of Strings] Ordered moves
        """
        pv_node_moves = []
        cut_node_moves = []
        all_node_moves = []
        leaf_node_moves = []
        killer_moves = []  # Children that are "killer" moves
        other_moves = []

        for move in node.child_moves:
            child = node.children[move]

            # If max depth reached and have exact score then use that
            if self.depth_limit == child.depth:
                result = self.tt.fetch(child.fen, 0)
                if result:
                    leaf_node_moves.append((move, result.value))
                    continue

            # Otherwise use best estimate if possible
            result = self.tt.best_estimate(child.fen)
            if result:
                move_inf = (move, result.value)
                if result.node_type == PV_NODE:
                    pv_node_moves.append(move_inf)
                elif result.node_type == CUT_NODE:
                    cut_node_moves.append(move_inf)
                elif result.node_type == LEAF_NODE:
                    leaf_node_moves.append(move_inf)
                elif self.is_killer(move, node.depth):
                    # Favor killer moves above all node moves
                    killer_moves.append(move_inf)
                else:
                    all_node_moves.append(move_inf)
            else:
                move_inf = (move, self.order_fn_fast(child.fen))
                if self.is_killer(move, node.depth):
                    killer_moves.append(move_inf)
                else:
                    other_moves.append(move_inf)

        # Order in ascending order (want to check boards which are BAD for opponent first
        pv_node_moves.sort(key=lambda x: x[1])
        cut_node_moves.sort(key=lambda x: x[1])
        all_node_moves.sort(key=lambda x: x[1])
        leaf_node_moves.sort(key=lambda x: x[1])
        killer_moves.sort(key=lambda x: x[1])
        other_moves.sort(key=lambda x: x[1])

        # If child has met depth, then favor leaf nodes the most
        #   Otherwise they are favored right above killer moves
        if self.depth_limit == (node.depth + 1):
            moves = leaf_node_moves + pv_node_moves + cut_node_moves + killer_moves + all_node_moves + other_moves
        else:
            moves = pv_node_moves + cut_node_moves + leaf_node_moves + killer_moves + all_node_moves + other_moves

        move_order = [x[0] for x in moves]

        return move_order

    def update_killer(self, killer_move, value, depth):
        """
        Updates the killer move table.
        Input:
            killer_move [Chess.Move]
                The move which caused the A-B pruning to trigger.
            value [Float]
                The value yielded by the killer move.
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
            self._add_killer_move(depth, value, killer_move)
        else:
            # Full
            # Check if move is better than worst current move
            if value > k_tab['values'][0]:
                # Remove Worst
                _, worst_move = k_tab['values'].pop(0)
                k_tab['moves'].remove(worst_move)

                # Add Item
                self._add_killer_move(depth, value, killer_move)

    def _add_killer_move(self, depth, value, killer_move):
        """
        Adds killer move to the table.
        """
        # Update moves
        self.killer_table[depth]['moves'].add(killer_move)

        # Update values
        self.killer_table[depth]['values'].append((value, killer_move))
        self.killer_table[depth]['values'].sort(key=lambda x: x[0])  # Sorting each time is OK since length is small.

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
        return move in self.killer_table[depth]['moves']

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

    def run(self, board, time_limit=None, reset=False):
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
                reset [bool]:
                    Resets the search instance. Used for training.
            Outputs:
                best_value [float]:
                    value achieved by best move
                best_move [chess.Move]:
                    Best move to play
                best_leaf [String]
                    FEN of the board of the leaf node which yielded the highest value.
        """
        if time_limit is None:
            time_limit = self.time_limit

        if reset:
            self.reset()
        self.num_evals = 0
        self.eval_time = 0
        self.num_visits = []

        # Start timing
        if time_limit is not None:
            self.time_limit = time_limit
        self.start_time = time.time()
        self.time_left = True

        self.root = SearchNode(board.fen(), 0)

        self.depth_limit = 1
        value = best_move = leaf_board = None
        while self.time_left and (self.max_depth is None or self.depth_limit <= self.max_depth):

            # Run search
            new_results = self.DLS(self.root, alpha=float("-inf"), beta=float("inf"))
            if not self.is_partial_search or (self.is_partial_search and self.use_partial_search):
                value, best_move, leaf_board = new_results
            self.is_partial_search = False

            # Prune if necessary
            if self.h_prune:
                self.prune(self.root)

            # Increase depth
            self.depth_limit += 1

        return value, best_move, leaf_board


class SearchNode:
    def __init__(self, fen, depth, value=None):
        """
        Generic node used for searching in 'rank_prune'.
        Input:
            fen [String]
                FEN of the board.
            depth [Int]
                Depth at which the node occurs.
            value [Float] (Optional)
                Value of the node.
        Non-Input Class Attributes
            children [Dict of SearchNode's]
                Children of the current SearchNode, key is Move which yields that SearchNode.
        """
        self.fen = fen
        self.depth = depth
        self.children = {}
        self.child_moves = []
        self.expand = True
        self.visited = False

    def _add_child(self, move, child):
        assert (isinstance(child, SearchNode))

        # Convert move to uci
        uci_move = move.uci()

        self.children[uci_move] = child
        self.child_moves.append(uci_move)

    def gen_children(self):
        # Create children
        board = chess.Board(self.fen)
        for move in board.legal_moves:
            board.push(move)
            self._add_child(move, SearchNode(board.fen(), self.depth + 1))
            board.pop()

    def get_child_nodes(self):
        """
        Returns a list of child nodes.
        """
        return self.children.values()

    def __str__(self):
        return "Node[{}, {}, {} children]".format(self.fen, self.depth, len(self.children))


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

        # Set depth limit to max depth
        self.depth_limit = max_depth

        # Turn off propagation
        self.use_prop = False

        # No buffer time needed
        self.buff_time = 0

    def __str__(self):
        return "Minimax"

    def run(self, board, time_limit=None, reset=False):
        """ Reset some variables, call recursive function """
        if reset:
            self.reset()
        self.num_evals = 0
        self.eval_time = 0
        self.num_visits = []
        self.time_left = True

        # Run to depth limit
        self.start_time = 0
        self.root = SearchNode(board.fen(), 0)  # Note: store unflipped fen
        return self.DLS(self.root, alpha=float("-inf"), beta=float("inf"))
