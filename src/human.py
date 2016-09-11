import player
import chess
import random

class Human(player.Player):
    def __init__(self, name, colour=None):
        super(Human, self).__init__(name, colour)

    def get_move(self, board):
        """
        Fetches a move from the command line.
        Input:
            Board [Chess.board]
        """

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
                print "Invalid or illegal input, legal moves are: "+ ', '.join([str(x) for x in board.legal_moves])

        return move
