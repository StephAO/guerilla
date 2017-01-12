from abc import ABCMeta, abstractmethod, abstractproperty
from guerilla.play.neural_net import NeuralNet
from guerilla.play.search import Search
import chess
import random
import os
import sys
import chess.uci
import other_engines.sunfish.sunfish as sunfish
import other_engines.sunfish.tools as sun_tools


class Player:
    __metaclass__ = ABCMeta

    def __init__(self, name, colour=None):
        """
        Initializes the Player abstract class.
        Input:
            name [String]
                Name of the player.
            colour [String]
                Colour of the player. White or Black.
        """
        self.name = name
        self._colour = colour

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
    def __init__(self, name, colour=None, load_file=None, hp_load_file=None, 
                 nn_input_type='bitmap', use_conv=True, num_fc=3, 
                 num_hidden=1024, num_feat=10, num_channels=12,
                 verbose=True):
        super(Guerilla, self).__init__(name, colour)
        self.nn = NeuralNet(load_file=load_file, hp_load_file=hp_load_file, 
                            nn_input_type=nn_input_type, use_conv=use_conv, 
                            num_fc=num_fc, num_hidden=num_hidden, 
                            num_feat=num_feat, num_channels=num_channels,
                            verbose=verbose)
        self.search = Search(self.nn.evaluate)

    def __enter__(self):
        self.nn.start_session()
        self.nn.init_graph()
        return self

    def __exit__(self, e_type, value, traceback):
        if e_type is not None:
            print e_type, value, traceback
        self.nn.close_session()

    def get_move(self, board):
        # print "Guerilla is thinking..."
        return self.search.run(board)[1]


class Human(Player):
    def __init__(self, name, colour=None):
        super(Human, self).__init__(name, colour)
        self.gui = None

    def get_move_from_gui(self, board):
        if self.gui is None:
            raise Exception("No gui was provided to human")

        move = self.gui.get_player_input(board, )
        print move
        return chess.Move.from_uci(move)

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

    def get_move(self, board, gui=True):
        """
        Fetches a move from the command line.
        Input:
            Board [Chess.board]
        """

        if gui:
            return self.get_move_from_gui(board)
        else:
            return self.get_move_from_tml(board)


class Sunfish(Player):
    def __init__(self, name, colour=None, time_limit=2):
        """
        Initializes the Sunfish class.
        Input:
            time_limit [Integer]
                The time limit for each move search in SECONDS.
        """
        super(Sunfish, self).__init__(name, colour)
        self.search = sunfish.Searcher()
        self.time_limit = time_limit

    def get_move(self, board):
        # Convert to Sunfish position
        sun_pos = sun_tools.parseFEN(board.fen())

        # Returns move
        move, _ = self.search.search(sun_pos, self.time_limit)
        return board.parse_san(sun_tools.renderSAN(sun_pos, move))


class Stockfish(Player):
    def __init__(self, name, colour=None, time_limit=2):
        """
        Initializes the Stockfish class.
        Input:
            time_limit [Integer]
                The time limit for each move search in SECONDS.
        """
        super(Stockfish, self).__init__(name, colour)
        self.time_limit = time_limit * 1000  # convert to milliseconds

        # Setup chess engine
        self.engine = chess.uci.popen_engine('stockfish')
        self.engine.uci()
        self.new_game()

    def get_move(self, board):
        self.engine.position(board)
        move, _ = self.engine.go(movetime=self.time_limit)
        return move

    def new_game(self):
        self.engine.ucinewgame()

    pass
