import sys
import chess
import player
import guerilla
import human

class Game:
    
    player_types = {
        'guerilla' : guerilla.Guerilla,
        'human' : human.Human
    }

    def __init__(self, p1, p2):
        """ 
            Note: p1 is white, p2 is black
            Input:
                player_1/player_2 [dict]:
                    type [string]: type of player (guerrilla, human, internet...)
                    name [string]: player name
                    kw_args [dict]: specific player type traits
        """
        if p1['type'] not in player_types or player_2['type'] not in player_types:
            print 'Error: player type is not supported'
            sys.exit()

        self.player1 = player_types[p1['type']](p1['name'])
        self.player2 = player_types[p2['type']](p2['name'])

        self.board = chess.Board()

    def play(self):
        white = True
        while not is_game_over(claim_draw=True):
            print self.board
            move = self.player1.get_move() if white else self.player2.get_move()
            while move not in self.board.legal_moves:
                print "Error: Move is not legal"
                move = self.player1.get_move() if white else self.player2.get_move()
            self.board.push(move)

def main():
    num_inputs = len(sys.argv)
    if num_inputs < 5:
        print "Error: not enough inputs. %d given, minimum 5 required" % (num_inputs)
        sys.exit()

    p1 = {
        'type' : sys.argv[1],
        'name' : sys.argv[2]
    }
    p2 = {
        'type' : sys.argv[3],
        'name' : sys.argv[4]
    }

if __name__ == '__main__':
    main()