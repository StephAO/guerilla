import chess
import random
import Queue
import time

class Node:
    """ 
        Node for monte carlo search tree. Each node must have a parent except the root node whose parent is None
    """
    root = None

    def __init__(self, parent, fen, depth):
        if parent is None and depth == 0:
            Node.root = self
        elif parent is None:
            raise RuntimeError("Every non-root node must have a parent")
        elif depth == 0:
            raise RuntimeError("Only root node may have a depth of 0")
        self.parent = parent
        self.board = chess.Board(fen) # Not sure if this is the most efficient way to do this
        self.moves = list(self.board.legal_moves)
        self.children = []
        self.depth = depth
        self.wins = 0.
        self.simulated_games = 0
        self.new_wins = 0.
        self.new_simulated_games = 0

    @staticmethod
    def select():
        """
            Uses a breadth first search to visit all the nodes and select the one with 
            the highest win/simulated_games ratio.
            # TODO use a better heuristic to select
        """
        selected_node = None
        nodes_to_vist = Queue.Queue()
        nodes_to_vist.put(Node.root)
        while not nodes_to_vist.empty():
            node = nodes_to_vist.get()
            print node.depth
            if node.moves and (selected_node is None or 
                               node.wins/node.simulated_games > selected_node.wins/selected_node.simulated_games):
                selected_node = node
            [nodes_to_vist.put(child) for child in node.children]
        return selected_node

    def expand(self, num_children=1):
        """
            Randomly generates a number of children. Ensures that no duplicate children are created
            Input:
                num_children[int]:
                    number of children to generate
            # TODO generate smartly if possible
        """
        new_children = []
        if num_children == -1:
            num_children = len(self.moves)
        for i in xrange(num_children):
            move = random.choice(self.moves)
            self.moves.remove(move)
            self.board.push(move)
            child = Node(self, self.board.fen(), self.depth + 1)
            new_children.append(child)
            self.board.pop()
        self.children.extend(new_children)
        return new_children

    def simulate(self, num_simulations=1):
        """
            Randomly plays a game and updates the nodes values based on the outcome
            # TODO make this better ;)
        """
        self.new_simulated_games += num_simulations
        for i in xrange(num_simulations):
            b = chess.Board(self.board.fen())
            while not b.is_game_over(claim_draw=True):
                
                # Get move
                move = random.choice(list(b.legal_moves))
                b.push(move)
                
            result = b.result(claim_draw=True)
            if result == '1-0':
                if Node.root.board.turn:
                    print '!'
                    self.new_wins += 1.
            elif result == '0-1':
                if not Node.root.board.turn:
                    print ';'
                    self.new_wins += 1.
            else:
                print '?'
                self.new_wins += 0.5

    def backpropagate(self):
        """ 
            Back propogate the results of the simulation. Current implementation is
            for the current simulation type. It will have to change according to the
            value that is backpropagated.
        """
        node = self
        while node:
            node.parent.wins += self.new_wins
            node.parent.simulated_games += self.new_simulated_games
            node = node.parent
        self.wins += self.new_wins
        self.simulated_games += self.new_simulated_games
        self.new_wins = 0
        self.new_simulated_games = 0

    def __str__(self):
        return str(self.board.fen())

    def __repr__(self):
        return str(self.depth)