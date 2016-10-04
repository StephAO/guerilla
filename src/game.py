import sys
import chess
import player
import guerilla
import human
import os
import time

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path + '/GUI/')
import ChessGUI


class Game:
    player_types = {
        'guerilla': guerilla.Guerilla,
        'human': human.Human
    }

    def __init__(self, players, num_games=1, use_gui=True):
        """ 
            Note: p1 is white, p2 is black
            Input:
                [player1, player2] [Class that derives Abstract Player Class]
        """

        assert all(isinstance(p, player.Player) for p in players)

        # Initialize players
        self.player1 = players[0]
        self.player2 = players[1]
        self.player1.colour = 'white'
        self.player2.colour = 'black'

        # Initialize board
        self.board = chess.Board()

        # Initialize statistics
        self.num_games = num_games
        self.data = dict()
        self.data['wins'] = [0, 0]
        self.data['draws'] = 0

        self.use_gui = use_gui
        # Initialize gui
        if use_gui:
            self.gui = ChessGUI.ChessGUI()
            if type(players[0]) is human.Human:
                self.player1.gui = self.gui
            if type(players[1]) is human.Human:
                self.player2.gui = self.gui

    def swap_colours(self):
        if self.player1.colour == 'white' and self.player2.colour == 'black':
            self.player1.colour = 'black'
            self.player2.colour = 'white'
        elif self.player1.colour == 'black' and self.player2.colour == 'white':
            self.player1.colour = 'white'
            self.player2.colour = 'black'
        else:
            raise ValueError('Error: one of the players has an invalid colour. ' + \
                 'Player 1: %s, Player 2: %s' % (self.player1.colour, self.player2.colour))

    def start(self):
        """ 
            Run n games. For each game players take turns until game is over.
            Note: draws are claimed automatically asap
        """

        for game in xrange(self.num_games):
            # Print info.
            print "Game %d - %s [%s] (%s) VS: %s [%s] (%s)" % (game, self.player1.name,
                                                                     type(self.player1).__name__,
                                                                     self.player1.colour,
                                                                     self.player2.name,
                                                                     type(self.player2).__name__,
                                                                     self.player2.colour)
            if self.use_gui:
                self.gui.print_msg("Game %d:" % (game))
                self.gui.print_msg("%s [%s] (%s)" % (self.player1.name, 
                                    type(self.player1).__name__, self.player1.colour))
                self.gui.print_msg("VS:")
                self.gui.print_msg("%s [%s] (%s)" % (self.player2.name,
                                    type(self.player2).__name__, self.player2.colour))
            # Reset board
            self.board.reset()

            player1_turn = self.player1.colour == 'white'

            game_pgn = chess.pgn.Game()
            game_pgn.headers["White"] = self.player1.name if player1_turn else self.player2.name
            game_pgn.headers["Black"] = self.player2.name if player1_turn else self.player1.name
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
                move = self.player1.get_move(self.board) if player1_turn else self.player2.get_move(self.board)
                game_pgn.add_main_variation(move)
                game_pgn = game_pgn.variation(0)
                while move not in self.board.legal_moves:
                    if self.use_gui:
                        self.gui.print_msg("Error: Move is not legal, try again")
                    else:
                        print "Error: Move is not legal"
                    move = self.player1.get_move(self.board) if player1_turn else self.player2.get_move(self.board)
                self.board.push(move)

                # Switch sides
                player1_turn = not player1_turn

            if self.use_gui:
                self.gui.end_of_game = True
                self.gui.draw(self.board)
                
            result = self.board.result(claim_draw=True)
            if result == '1-0':
                self.data['wins'][0] += 1
                print "%s wins." % self.player1.name
                if self.use_gui:
                    self.gui.print_msg("%s wins." % self.player1.name)
            elif result == '0-1':
                self.data['wins'][1] += 1
                print "%s wins." % self.player2.name
                if self.use_gui:
                    self.gui.print_msg("%s wins." % self.player2.name)
            else:
                self.data['draws'] += 1
                print "Draw."
                if self.use_gui:
                    self.gui.print_msg("Draw.")

            game_pgn = game_pgn.root()
            game_pgn.headers["Result"] = result
            with open(dir_path + "/../played_games/" + self.player1.name + '_' + \
                      self.player2.name + '_' + str(game) + '.pgn', 'w') as pgn:
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

        players[0] = guerilla.Guerilla('Harambe (selfplay)', _load_file='in_training_weight_values.p')
        players[1] = guerilla.Guerilla('Donkey Kong (selfplay)', _load_file='in_training_weight_values.p')

        # players[0] = human.Human("A")
        # players[1] = human.Human("B")

    elif choose_players == 'c':
        for i in xrange(2):
            print "Player 1 will start as white the first game. Each game players will swap colours"
            player_name = raw_input("Player %d name: " % (i))
            player_type = raw_input("Player %d type %s : " % (i, Game.player_types.keys()))
            if player_type == 'guerilla':
                weight_file = raw_input("Load_file or (d) for default. (File must be located in the pickles directory):\n")
                players[i] = guerilla.Guerilla(player_name, _load_file=(weight_file if weight_file!='d' else None))
            elif player_type == 'human':
                players[i] = human.Human(player_name)
            else:
                raise NotImplementedError("Player type selected is not supported. See README.md for player types")

    game = Game(players, num_games=5)
    if isinstance(players[0], guerilla.Guerilla) and isinstance(players[1], guerilla.Guerilla):
        with players[0], players[1]:
            game.start()
    elif isinstance(players[0], guerilla.Guerilla):
        with players[0]:
            game.start()
    elif isinstance(players[1], guerilla.Guerilla):
        with players[1]:
            game.start()
    else:
        game.start()

if __name__ == '__main__':
    main()
