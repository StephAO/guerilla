import sys
import chess
import chess.pgn
from guerilla.players import *
import os
import time
from pkg_resources import resource_filename
from guerilla.play.gui.chess_gui import ChessGUI


class Game:
    player_types = {
        'guerilla': Guerilla,
        'human': Human,
        'sunfish': Sunfish,
        'stockfish': Stockfish
    }

    def __init__(self, players, num_games=1, use_gui=True):
        """ 
            Note: p1 is white, p2 is black
            Input:
                [player1, player2] [Class that derives Abstract Player Class]
        """

        assert all(isinstance(p, Player) for p in players)

        # Initialize players
        self.players = players
        self.players[0].colour = 'white'
        self.players[1].colour = 'black'

        # Initialize board
        self.board = chess.Board()

        # Initialize statistics
        self.num_games = num_games
        self.data = dict()
        self.data['wins'] = [0, 0]
        self.data['draws'] = 0

        self.use_gui = use_gui or any(isinstance(p, Human) for p in players)
        # Initialize gui
        if use_gui:
            self.gui = ChessGUI()
            for p in players:
                if isinstance(p, Human):
                    p.gui = self.gui

    def swap_colours(self):
        # Ensure that there is currently a white player and a black player
        if all(colour in [p.colour for p in self.players] for colour in ['white', 'black']):
            self.players[0].colour, self.players[1].colour = self.players[1].colour, self.players[0].colour
        else:
            raise ValueError('Error: one of the players has an invalid colour. ' +
                             'Player 1: %s, Player 2: %s' % (self.players[0].colour, self.players[1].colour))

    def start(self):
        """ 
            Run n games. For each game players take turns until game is over.
            Note: draws are claimed automatically asap
        """
        with self.players[0], self.players[1]:

            for game in xrange(self.num_games):

                # Print info.
                print "Game %d - %s [%s] (%s) VS: %s [%s] (%s)" % (game + 1, self.players[0].name,
                                                                   type(self.players[0]).__name__,
                                                                   self.players[0].colour,
                                                                   self.players[1].name,
                                                                   type(self.players[1]).__name__,
                                                                   self.players[1].colour)
                if self.use_gui:
                    self.gui.print_msg("Game %d:" % (game + 1))
                    self.gui.print_msg("%s [%s] (%s)" % (self.players[0].name,
                                                         type(self.players[0]).__name__, self.players[0].colour))
                    self.gui.print_msg("VS:")
                    self.gui.print_msg("%s [%s] (%s)" % (self.players[1].name,
                                                         type(self.players[1]).__name__, self.players[1].colour))
                # Reset board
                self.board.reset()

                # Signal to players that a new game is being played.
                [p.new_game() for p in self.players]

                player1_turn = self.players[0].colour == 'white'
                curr_player_idx = 0 if player1_turn else 0

                game_pgn = chess.pgn.Game()
                game_pgn.headers["White"] = self.players[curr_player_idx].name
                game_pgn.headers["Black"] = self.players[(curr_player_idx + 1) % 2].name
                game_pgn.headers["Date"] = time.strftime("%Y.%m.%d")
                game_pgn.headers["Event"] = "Test"
                game_pgn.headers["Round"] = game
                game_pgn.headers["Site"] = "My PC"

                # Start game
                while not self.board.is_game_over(claim_draw=True):
                    if self.use_gui:
                        self.gui.draw(self.board)
                    else:
                        Game.pretty_print_board(self.board)

                    # Get move
                    st = time.time()
                    move = self.players[curr_player_idx].get_move(self.board)
                    self.players[curr_player_idx].time_taken += time.time() - st
                    game_pgn.add_main_variation(move)
                    game_pgn = game_pgn.variation(0)
                    while move not in self.board.legal_moves:
                        if self.use_gui:
                            self.gui.print_msg("Error: Move is not legal, try again")
                        else:
                            print "Error: Move is not legal"
                        move = self.players[curr_player_idx].get_move(self.board)
                    self.board.push(move)
                    if self.use_gui:
                        self.gui.print_msg("%s played %s" % (self.players[curr_player_idx].name, move))
                    else:
                        print "%s played %s" % (self.players[curr_player_idx].name, move)

                    # Switch sides
                    curr_player_idx = (curr_player_idx + 1) % 2

                if self.use_gui:
                    self.gui.end_of_game = True
                    self.gui.draw(self.board)

                result = self.board.result(claim_draw=True)
                if result == '1-0':
                    winner = self.players[0] if self.players[0].colour == 'white' else self.players[1]
                elif result == '0-1':
                    winner = self.players[1] if self.players[0].colour == 'white' else self.players[0]
                else:
                    winner = None
                    self.data['draws'] += 1
                    if self.use_gui:
                        self.gui.print_msg("Draw.")
                    else:
                        print "Draw."

                if winner is not None:
                    winner_idx = (curr_player_idx + 1) % 2
                    self.data['wins'][winner_idx] += 1
                    if self.use_gui:
                        self.gui.print_msg("%s wins." % winner.name)
                    else:
                        print "%s wins." % winner.name

                for p in self.players:
                    print "Player %s took %f seconds in total" % (p.name, p.time_taken)
                    p.time_taken = 0

                game_pgn = game_pgn.root()
                game_pgn.headers["Result"] = result
                with open(resource_filename('guerilla', 'data/played_games/') + self.players[0].name + '_' +
                            self.players[1].name + '_' + str(game) + '.pgn', 'w') as pgn:
                    pgn.write(str(game_pgn))

                if self.use_gui:
                    self.gui.wait_for_endgame_input()

                self.swap_colours()

    @staticmethod
    def pretty_print_board(board):
        """
        Prints the chess board with grid annotations.
        Input:
            Board [Chess.board]
        """
        board_rows = str(board).split('\n')
        annotated = [str(8 - i) + "\t" + x for i, x in enumerate(board_rows)]
        print '\n'.join(annotated)
        print '\n\ta b c d e f g h'

        return

        # @staticmethod
        # def get_gui_board_representation(fen):
        #     board = [[]]
        #     for i, char in enumerate(fen.split()[0]):
        #         if char == '/':
        #             board.append([])
        #             i += 1
        #         elif char.isdigit():
        #             for j in xrange(int(char)):
        #                 board[i].append('e')
        #         else:
        #             prefix = 'w' if char.isupper() else 'b'


def main():
    num_inputs = len(sys.argv)
    choose_players = raw_input("Choose players (c) or use defaults (d): ")
    players = [None] * 2
    if choose_players == 'd':

        # players[1] = Guerilla('Harambe (COMPLEMENTMAX)', search_type='complementmax', _load_file='w_train_bootstrap_0212-0248_movemap_3FC.p')
        # players[0] = Guerilla('Donkey Kong (full)', _load_file='weights_train_td_endgames_20161006-065100.p')

        # players[1] = Sunfish("Sun", time_limit=1)
        # players[0] = Guerilla('King Kong (RANKPRUNE)', search_type='rankprune', load_file='w_train_bootstrap_0212-0248_movemap_3FC.p')
        # 
        players[0] = Human('a')
        players[1] = Human('b')

        # players[1].search.max_depth = 3
        # players[0].search.max_depth = 2
        # players[1].search.max_depth = 2


    elif choose_players == 'c':
        for i in xrange(2):
            print "Player 1 will start as white the first game. Each game players will swap colours"
            player_name = raw_input("Player %d name: " % (i))
            player_type = raw_input("Player %d type %s : " % (i, Game.player_types.keys()))
            if player_type == 'guerilla':
                weight_file = raw_input(
                    "Load_file or (d) for default. (File must be located in the pickles directory):\n")
                players[i] = Guerilla(player_name, load_file=(weight_file if weight_file != 'd' else None))
            elif player_type == 'human':
                players[i] = Human(player_name)
            else:
                raise NotImplementedError("Player type selected is not supported. See README.md for player types")

    game = Game(players, num_games=2)
    # if isinstance(players[0], Guerilla) and isinstance(players[1], Guerilla):
    #     with players[0], players[1]:
    #         game.start()
    # elif isinstance(players[0], Guerilla):
    #     with players[0]:
    #         game.start()
    # elif isinstance(players[1], Guerilla):
    #     with players[1]:
    #         game.start()
    # else:
    game.start()


if __name__ == '__main__':
    main()
