from abc import ABCMeta, abstractmethod
import chess
import math
import time
import numpy as np
from collections import namedtuple

from search_helpers import quickselect, material_balance
import guerilla.data_handler as dh

# TODO: Deal with node_type
# Note:
#   Moves are stored as UCI strings
#   Leaf FEN is stripped
Transposition = namedtuple("Transposition", "best_move score leaf_fen node_type")

PV_NODE = 0
CUT_NODE = 1
ALL_NODE = 2


class TranspositionTable:
    def __init__(self, exact_depth=True):
        self.table = {}  # Main transposition table {FEN: Transposition Entry}
        self.exact_depth = exact_depth  # If we require an exact depth match

        # TODO
        self.cache_hits = 0
        self.cache_miss = 0

    def fetch(self, fen, requested_depth):
        # Returns {Best Move, Score, Type, Depth}

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
        if requested_depth in entry.score_dict:
            self.cache_hits += 1
            transpo = entry.score_dict[requested_depth]

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
        return Transposition(best_move, entry.score, leaf_fen, entry.node_type)


class TranpositionEntry:
    def __init__(self):
        self.score_dict = {}  # {Depth: {Best Move (UCI), Score, Leaf FEN, Node Type,}}
        self.deepest = None

    def add_depth(self, depth, transposition):
        # Update Score dict and deepest
        if depth in self.score_dict:
            raise ValueError("Should never try to update an existing depth!")
        self.score_dict[depth] = transposition

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

    def evaluate(self, fen, depth):
        # Depth is used to update information

        # Track some info
        try:
            self.num_visits[depth] += 1
        except IndexError:
            self.num_visits.append(1)

        # check if new depth reached
        if depth > self.depth_reached:
            self.depth_reached = depth

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
                best_score [float]:
                    Score achieved by best move
                best_leaf [String]
                    FEN of the board of the leaf node which yielded the highest value.
        """
        raise NotImplementedError("You should never see this")


    def clear_cache(self):
        # Clears the transposition table
        self.tt.table.clear()
        self.tt.cache_hits = self.tt.cache_miss = 0
        # TODO


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
        self.order_fn_fast = material_balance  # The move ordering function to use pre-Guerilla evaluation # TODO: Replace w/ lower depth scores
        self.is_partial_search = False  # Marks if the last DFS call was a partial search
        self.use_partial_search = use_partial_search

        # Holds the Killer Moves by depth. Each Entry is (set of moves, sorted array of (score, moves)).
        self.killer_table = [{'moves': set(), 'scores': list()}]
        self.num_killer = 2  # Number of killer moves store for each depth

    def __str__(self):
        return "IterativeDeepening"

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
        leaf_fen = node.fen
        node_type = PV_NODE

        result = self.tt.fetch(node.fen, requested_depth=self.depth_limit - node.depth)
        if result and result.node_type == PV_NODE:
            # print "FOUND TRANSPOSITION! {}".format(node.fen)
            # TODO: For now ignore if not type == PV
            return result.score, result.best_move, result.leaf_fen

        # Create child nodes if not already done
        # TODO: This is here b/c of checkmates, maybe can avoid generating children and move back into else
        if not node.children:
            node.gen_children()

        self.time_left = (time.time() - self.start_time) <= self.time_limit - self.buff_time
        # Evaluate IF: depth limit reached, pruned, no time left OR no children
        if node.depth >= self.depth_limit or not node.expand or not self.time_left or not node.children:
            if not self.time_left:
                self.is_partial_search = True
            # Evaluate node
            node.value = self.evaluate(node.fen, node.depth)
        else:
            # Check if a new killer table entry should be created
            if node.depth >= len(self.killer_table):
                self.killer_table.append({'moves': set(), 'scores': list()})

            killers = []  # Children that are "killer" moves
            value_order = []  # Children w/ values
            fast_order = []  # Children w/o values

            # Accessing of children is done this way so that order is consistent
            move_order = node.child_moves
            if self.order_moves:
                for move in move_order:
                    child = node.children[move]
                    # Child has not previously been evaluated, use fast scoring & ordering
                    if child.value:
                        # Get scores
                        value = self.order_fn_fast(child.fen)
                    else:
                        # Child has previously been evaluated, use node evaluation for ordering
                        #   (This is possible because its iterative deepening)
                        value = child.value

                    # Check which ordering it should go into
                    move_inf = (move, value)
                    if self.is_killer(move, node.depth):
                        killers.append(move_inf)
                    elif child.value:
                        value_order.append(move_inf)
                    else:
                        fast_order.append(move_inf)

                # Order in ascending order (want to check boards which are BAD for opponent first)
                killers.sort(key=lambda x: x[1])
                value_order.sort(key=lambda x: x[1])
                fast_order.sort(key=lambda x: x[1])

                # Favor killer moves the most, then nodes with value, then others
                move_order = [x[0] for x in killers + value_order + fast_order]

            for move in move_order:
                # Do DLS on children
                # print "[{}] {}".format(node.depth, move)
                score, _, lf = self.DLS(node.children[move], parent=node, a=-best_score)



                score = -score
                # print "[{}] PARENT {} MOVE {} SCORE {}".format(node.depth, node.fen, move, score)
                # Best child is the one one which has the lowest value
                if best_move is None or score > best_score:
                    best_score = score
                    best_move = move
                    leaf_fen = lf

                if self.ab_prune and best_score >= a:
                    # print "PRUNED @ {} AFTER CHECKING {}".format(node.fen, move)
                    self.update_killer(move, best_score, node.depth)
                    # TODO: HANDLE MOVE TYPE
                    node.type = CUT_NODE
                    break

            node.value = best_score

        # Update transposition table
        # print "ADDING {}".format(node)
        self.tt.update(node.fen, self.depth_limit - node.depth, best_move, node.value, leaf_fen, node_type)

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
        self.killer_table[depth]['moves'].add(killer_move)

        # Update scores
        self.killer_table[depth]['scores'].append((score, killer_move))
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

        self.root = SearchNode(board.fen(), 0, self.evaluate(board.fen(), 0))

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
        self.value = value
        self.children = {}
        self.child_moves = []
        self.expand = True

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
        return "Node[{}, {}, {}, {} children]".format(self.fen, self.depth, self.value, len(self.children))


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
        self.root = SearchNode(board.fen(), 0)  # Note: store unflipped fen
        return self.DLS(self.root)
