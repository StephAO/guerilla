import sys
import chess
import player
import guerilla
import human
import os

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
                [player_1, player_2] [Class that derives Abstract Player Class]
        """
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

        # Initialize gui
        if use_gui:
            self.gui = ChessGUI.ChessGUI()
            if type(p1) is human.Human:
                self.player1.gui = self.gui
            if type(p2) is human.Human:
                self.player2.gui = self.gui

    def swap_colours(self):
        if self.player1 == 'white' and self.player2 == 'black':
            self.player1.colour = 'black'
            self.player2.colour = 'white'
        elif self.player1 == 'black' and self.player2 == 'white':
            self.player1.colour = 'white'
            self.player2.colour = 'black'
        else:
            raise ValueError('Error: one of the players has an invalid colour')

    def start(self):
        """ 
            Run n games. For each game players take turns until game is over.
            Note: draws are claimed automatically asap
        """

        for game in xrange(self.num_games):
            # Print info.
            print "Game %d - %s [%s] (White) VS: %s [%s] (Black)" % (game, self.player1.name,
                                                                     type(self.player1).__name__,
                                                                     self.player2.name,
                                                                     type(self.player2).__name__)

            # Reset board
            self.board.reset()

            # Start game
            white = True
            while not self.board.is_game_over(claim_draw=True):
                if use_gui:
                    self.gui.draw(self.board)
                else:
                    Game.pretty_print_board(self.board)
                
                print self.board.fen()

                # Get move
                move = self.player1.get_move(self.board) if white else self.player2.get_move(self.board)
                print move
                while move not in self.board.legal_moves:
                    if use_gui:
                        self.gui.print_msg("Error: Move is not legal, try again")
                    else:
                        print "Error: Move is not legal"
                    move = self.player1.get_move(self.board) if white else self.player2.get_move(self.board)
                self.board.push(move)

                # Switch sides
                white = not white

            result = self.board.result(claim_draw=True)
            if result == '1-0':
                self.data['wins'][0] += 1
                print "%s wins." % self.player1.name
                if use_gui:
                    self.gui.print_msg("%s wins." % self.player1.name)
            elif result == '0-1':
                self.data['wins'][1] += 1
                print "%s wins." % self.player2.name
                if use_gui:
                    self.gui.print_msg("%s wins." % self.player2.name)
            else:
                self.data['draws'] += 1
                print "Draw."
                if use_gui:
                    self.gui.print_msg("Draw.")
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

        players[0] = guerilla.Guerilla('Harambe', _load_file='weight_values.p')
        players[1] = human.Human('Cincinnati Zoo')

    elif choose_players == 'c':
        for i in xrange(2):
            print "Player 1 will start as white the first game. Each game players will swap colours"
            player_name = raw_input("Player %d name: " % (i))
            player_type = raw_input("Player %d type %s : " % (i, Game.player_types.keys()))
            if player_type == 'guerilla':
                weight_file = raw_input("Load_file or (d) for default. (File must be located in the pickles directory):\n") 
                players[i] = guerilla.Guerilla(name, _load_file=weight_file)
            elif player_type == 'human':
                players[i] = human.Human(player_name)
            else:
                raise NotImplementedError("Player type selected is not supported. See README.md for player types")

    game = Game(players)
    game.start()


if __name__ == '__main__':
    main()
