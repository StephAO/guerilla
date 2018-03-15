import chess
import random
import chess.uci
from abc import ABCMeta, abstractmethod

import guerilla.data_handler as dh
from guerilla.play.neural_net import NeuralNet
from guerilla.play.search import Minimax, IterativeDeepening

class Player:
    __metaclass__ = ABCMeta

    def __init__(self, name):
        """
        Initializes the Player abstract class.
        Input:
            name [String]
                Name of the player.
        """
        self.name = name

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, colour):
        if colour.lower() in ['white', 'w']:
            self._colour = 'white'
        elif colour.lower() in ['black', 'b']:
            self._colour = 'black'
        else:
            raise ValueError("Error: Invalid colour! Must be 'white','w','black', or 'b'.")

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError("You should never see this")

    @abstractmethod
    def __exit__(self, e_type, value, traceback):
        raise NotImplementedError("You should never see this")

    @abstractmethod
    def get_move(self, board):
        """
        Gets the next move from a player given the current board.
        Input:
            Board [chess.Board]
        Output:
            Move [chess.Move]
        """
        raise NotImplementedError("You should never see this")

    def new_game(self):
        """
        Let's the player know that a new game is being played.
        Output:
            None.
        """
        pass


class Guerilla(Player):
    search_types = {
        "minimax": Minimax,
        "iterativedeepening": IterativeDeepening
    }

    def __init__(self, name, colour=None, search_type='minimax', load_file=None,
                 hp_load_file=None, seed=None, verbose=True, nn_params=None, search_params=None):
        super(Guerilla, self).__init__(name)

        self.nn = NeuralNet(load_file=load_file, hp_load_file=hp_load_file, seed=seed,
                            verbose=verbose, hp=nn_params)

        search_params = {} if search_params is None else search_params

        self.search = Guerilla.search_types[search_type](self.nn.evaluate, **search_params)

    def __enter__(self):
        self.nn.start_session()
        self.nn.init_graph()
        return self

    def __exit__(self, e_type, value, traceback):
        if e_type is not None:
            print e_type, value, traceback
        self.nn.close_session()
        self.nn.reset_graph()

    def get_move(self, board):
        # print "Guerilla is thinking..."
        score, move, leaf = self.search.run(board)
        if move is None:
            raise ValueError("There are no valid moves from this position! FEN: %s "
                             "\n\t Debug Info: Score: %f Move: %s Leaf Board: %s" % (
                             board.fen(), score, str(move), leaf))
        return move

    def get_cp_adv_white(self, fen):
        """
        Returns the centipawn advantage of white given the current fen.
        Input:
            fen [String]
                FEN.
        Output:
            centipawn advantage [Float]
                Centipawn advantage of white.
        """
        if dh.white_is_next(fen):
            return self.nn.evaluate(fen)
        else:
            # Black plays next
            return -self.nn.evaluate(dh.flip_board(fen))

    def get_cp_adv_black(self, fen):
        """
        Returns the centipawn advantage of black given the current fen.
        Input:
            fen [String]
                FEN.
        Output:
            centipawn advantage [Float]
                Centipawn advantage of black.
        """
        return -self.get_cp_adv_white(fen)


class Human(Player):
    def __init__(self, name, colour=None):
        super(Human, self).__init__(name)

    def __enter__(self):
        return self

    def __exit__(self, e_type, value, traceback):
        if e_type is not None:
            print e_type, value, traceback

    def get_move_from_tml(self, board):
        move = None

        while True:
            print "Please enter your move %s (? for help):" % self.name
            usr_input = raw_input().lower()

            if usr_input == '?':
                print "Input move must be in algebraic notation (i.e. a2a3). \nOther commands"
                print "\t?\tDisplay help (this text).\n\tl\tPrint legal moves.\n\tr\tPlay a random (legal) move."
            elif usr_input == 'l':
                print "Legal Moves: " + ', '.join([str(x) for x in board.legal_moves])
            elif usr_input == 'r':
                move = random.sample(board.legal_moves, 1)[0]
                break
            elif usr_input in [str(x) for x in board.legal_moves]:
                move = chess.Move.from_uci(usr_input)
                break
            else:
                print "Invalid or illegal input, legal moves are: " + ', '.join([str(x) for x in board.legal_moves])

        return move

    def get_move(self, board):
        """
        Fetches a move from the command line.
        Input:
            Board [Chess.board]
        """
        return self.get_move_from_tml(board)


class Stockfish(Player):
    def __init__(self, name, colour=None, time_limit=2):
        """
        Initializes the Stockfish class.
        Input:
            time_limit [Integer]
                The time limit for each move search in SECONDS.
        """
        super(Stockfish, self).__init__(name)
        self.time_limit = time_limit * 1000  # convert to milliseconds

        # Setup chess engine
        self.engine = chess.uci.popen_engine('stockfish')
        self.engine.uci()
        self.new_game()

    def __enter__(self):
        return self

    def __exit__(self, e_type, value, traceback):
        if e_type is not None:
            print e_type, value, traceback

    def get_move(self, board):
        self.engine.position(board)
        move, _ = self.engine.go(movetime=self.time_limit)
        return move

    def new_game(self):
        self.engine.ucinewgame()

    pass


def main():
    # test
    with Guerilla('Harambe', search_type='iterativedeepening', search_params={'time_limit': 5},
                  load_file='6811.p') as g:
        board = chess.Board()
        print g.get_move(board)
        print "HIT: {} MISS: {} DEPTH REACHED: {}".format(g.search.tt.cache_hit, g.search.tt.cache_miss,
                                                          g.search.depth_reached)


if __name__ == '__main__':
    main()
