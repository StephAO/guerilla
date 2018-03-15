import os
import chess
import chess.pgn
import sys
import time

from pkg_resources import resource_filename
from guerilla.players import *


class Game:
    player_types = {
        'guerilla': Guerilla,
        'human': Human,
    }

    try:
        player_types['stockfish'] = Stockfish
    except NameError:
        pass

    def __init__(self, players, num_games=1):
        """ 
            Note: p1 is white, p2 is black
            Input:
                [player1, player2] [Class that derives Abstract Player Class]
        """
        assert all(isinstance(p, Player) for p in players.itervalues())

        # Initialize players
        self.players = players

        # Initialize board
        self.board = chess.Board()

        # Initialize statistics
        self.num_games = num_games
        self.data = dict()
        self.data['wins'] = {p.name: 0 for p in players.itervalues()}
        self.data['draws'] = 0


    def swap_colours(self):
        """ Swap colours."""
        self.players['w'], self.players['b'] = self.players['b'], self.players['w']

    def set_board(self, fen):
        self.board = chess.Board(fen)

    def play(self, curr_player, game_pgn=None, verbose=True, moves_left=-1):
        game_fens = [self.board.fen()]

        time_taken = {'w': 0, 'b': 0}

        # Start game
        while not self.board.is_game_over(claim_draw=True) and moves_left != 0:
            Game.pretty_print_board(self.board)

            # Get move
            st = time.time()
            move = self.players[curr_player].get_move(self.board)
            time_taken[curr_player] += time.time() - st

            # hack to go back in time
            if move == "undo":
                if self.board.move_stack:
                    self.board.pop()
                    self.board.pop()
                continue

            while move not in self.board.legal_moves:
                print "Error: Move is not legal"
                move = self.players[curr_player].get_move(self.board)
            self.board.push(move)
            print "%s played %s" % (self.players[curr_player].name, move)

            if game_pgn is not None:
                game_pgn.add_main_variation(move)
                game_pgn = game_pgn.variation(0)
            else:
                game_fens.append(self.board.fen())

            # Switch sides
            curr_player = 'w' if curr_player == 'b' else 'b'
            moves_left -= 1

        return game_fens, time_taken

    def start(self):
        """ 
            Run n games. For each game players take turns until game is over.
            Note: draws are claimed automatically asap
        """
        with self.players['w'], self.players['b']:

            game = 0

            while game < self.num_games:

                # Print info.
                print "Game %d - %s [%s] (White) VS: %s [%s] (Black)" % (game + 1,
                                                                         self.players['w'].name,
                                                                         type(self.players['w']).__name__,
                                                                         self.players['b'].name,
                                                                         type(self.players['b']).__name__)
                # Reset board
                self.board.reset()

                # Signal to players that a new game is being played.
                [p.new_game() for p in self.players.itervalues()]

                curr_player_idx = 'w'

                game_pgn = chess.pgn.Game()
                game_pgn.headers["White"] = self.players['w'].name
                game_pgn.headers["Black"] = self.players['b'].name
                game_pgn.headers["Date"] = time.strftime("%Y.%m.%d")
                game_pgn.headers["Event"] = "Test"
                game_pgn.headers["Round"] = game
                game_pgn.headers["Site"] = "My PC"

                _, time_taken = self.play(curr_player_idx, game_pgn=game_pgn)

                result = self.board.result(claim_draw=True)
                if result == '1-0':
                    winner = self.players['w']
                elif result == '0-1':
                    winner = self.players['b']
                else:
                    winner = None
                    self.data['draws'] += 1
                    print "Draw."                        

                if winner is not None:
                    self.data['wins'][winner.name] += 1
                    print "%s wins." % winner.name

                for color, p in self.players.iteritems():
                    print "Player %s took %f seconds in total" % (p.name, time_taken[color])
                    p.time_taken = 0

                game_pgn = game_pgn.root()
                game_pgn.headers["Result"] = result
                with open(resource_filename('guerilla', 'data/played_games/') + self.players['w'].name + '_' +
                                  self.players['b'].name + '_' + str(game) + '.pgn', 'w') as pgn:
                    try:
                        pgn.write(str(game_pgn))
                    except AttributeError as e:
                        print "Error writing pgn file: %s" % (e)

                self.swap_colours()
                game += 1

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
    choose_players = raw_input("Choose players (c) or use defaults (d): ")
    players = {'w': None, 'b': None}
    if choose_players == 'd':

        players['w'] = Stockfish('test', time_limit=1)
        players['b'] = Guerilla('Harambe', search_type='minimax', load_file='7034.p', search_params={'max_depth': 2})

    elif choose_players == 'c':
        for i in ['w', 'b']:
            print "Player 1 will start as white the first game. Each game players will swap colours"
            color = 'White' if i == 'w' else 'Black'
            player_name = raw_input("%s player name: " % (color))
            player_type = raw_input("%s player type %s : " % (color, Game.player_types.keys()))
            if player_type == 'guerilla':
                weight_file = raw_input(
                    "Load_file or (d) for default. (File must be located in the pickles directory):\n")
                players[i] = Guerilla(player_name, load_file=(weight_file if weight_file != 'd' else 'default.p'))
            elif player_type in Game.player_types.keys():
                players[i] = Game.player_types[player_type](player_name)
            else:
                raise NotImplementedError("Player type selected is not supported. See README.md for player types")

    game = Game(players)
    game.start()


if __name__ == '__main__':
    main()
