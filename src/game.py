import sys
import chess
import player
import guerilla
import human


class Game:
    player_types = {
        'guerilla': guerilla.Guerilla,
        'human': human.Human
    }

    def __init__(self, p1, p2, num_games=1):
        """ 
            Note: p1 is white, p2 is black
            Input:
                player_1/player_2 [Class that derives Abstract Player Class]
        """
        if type(p1) not in Game.player_types.values() or type(p2) not in Game.player_types.values():
            raise NotImplementedError("Player type selected is not supported. See README.md for player types")

        # Initialize players
        self.player1 = p1
        self.player2 = p2
        self.player1.colour = 'w'
        self.player2.colour = 'b'

        # Initialize board
        self.board = chess.Board()

       # Initialize statistics
        self.num_games = num_games
        self.data = {}
        self.data['wins'] = [0, 0]
        self.data['draws'] = 0

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
                Game.pretty_print_board(self.board)

                # Get move
                move = self.player1.get_move(self.board) if white else self.player2.get_move(self.board)
                while move not in self.board.legal_moves:
                    print "Error: Move is not legal"
                    move = self.player1.get_move(self.board) if white else self.player2.get_move(self.board)
                self.board.push(move)

                # Switch sides
                white = not white

            result = self.board.result(claim_draw=True)
            if result == '1-0':
                self.data['wins'][0] += 1
                print "%s wins." % self.player1.name
            elif result == '0-1':
                self.data['wins'][1] += 1
                print "%s wins." % self.player2.name
            else:
                self.data['draws'] += 1
                print "Draw."

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

def main():
    num_inputs = len(sys.argv)
    if num_inputs >= 5:
        print "Using inputs to start game"

        p1 = {
            'type': sys.argv[1],
            'name': sys.argv[2]
        }
        p2 = {
            'type': sys.argv[3],
            'name': sys.argv[4]
        }
    else:
        print "Under 5 (%d) inputs, using defaults to start game." % (num_inputs)
        p1 = {
            'type': 'guerilla',
            'name': 'Harambe'
        }
        p2 = {
            'type': 'human',
            'name': 'Cincinnati Zoo'
        }

    if (p1['type'] not in Game.player_types) or (p2['type'] not in Game.player_types):
        print "Error: Player type selected is not supported. See README.md for player types."
        return

    # Create classes based on inputs.
    player1 = Game.player_types[p1['type']](p1['name'], 'weight_values.p')
    player2 = Game.player_types[p2['type']](p2['name'])

    game = Game(player1, player2)
    game.start()


if __name__ == '__main__':
    main()
