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

    def __init__(self, p1, p2, num_games=1):
        """ 
            Note: p1 is white, p2 is black
            Input:
                player_1/player_2 [dict]:
                    type [string]: type of player (guerrilla, human, internet...)
                    name [string]: player name
        """
        if p1['type'] not in Game.player_types or p2['type'] not in Game.player_types:
            raise NotImplementedError("Player type selected is not supported. See README.md for player types")

        self.player1 = Game.player_types[p1['type']](p1['name'], 'white')
        self.player2 = Game.player_types[p2['type']](p2['name'], 'black')

        self.board = chess.Board()

        self.data = {}
        self.data['wins'] = [0,0]
        self.data['draws'] = 0

    def start(self):
        """ 
            Run n games. For each game players take turns until game is over.
            Note: draws are claimed automatically asap
        """
        for game in xrange(self.num_games):
            white = True
            while not is_game_over(claim_draw=True):
                print self.board
                move = self.player1.get_move() if white else self.player2.get_move()
                while move not in self.board.legal_moves:
                    print "Error: Move is not legal"
                    move = self.player1.get_move() if white else self.player2.get_move()
                self.board.push(move)
            result = self.board.result(claim_draw=True)
            if result == '1-0':
                self.data['wins'][0] += 1
            elif result == '0-1':
                self.data['wins'][1] += 1
            else:
                self.data['draws'] += 1

def main():
    num_inputs = len(sys.argv)
    if num_inputs >= 5:
        print "Using inputs to start game"

        p1 = {
            'type' : sys.argv[1],
            'name' : sys.argv[2]
        }
        p2 = {
            'type' : sys.argv[3],
            'name' : sys.argv[4]
        }
    else:
        print "Under 5 (%d) inputs, using defaults to start game." % (num_inputs)
        p1 = {
            'type' : 'guerilla',
            'name' : 'Harambe'
        }
        p2 = {
            'type' : 'human',
            'name' : 'Cincinnati Zoo'
        }

    game = Game(p1,p2)
    game.start()


if __name__ == '__main__':
    main()