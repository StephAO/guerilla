import chess.pgn
import sys
import os

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, dir_path + '/../src/GUI/')
import ChessGUI

def view_game(filename):
    gui = ChessGUI.ChessGUI(view=True)
    full_path = dir_path + "/../played_games/" + filename
    gui.print_msg("Use arrow keys or buttons to navigate")
    with open(full_path, 'r') as pgn_file:
        game = chess.pgn.read_game(pgn_file)
        while True:
            gui.draw(game.board())
            next_move = gui.wait_for_view_input()
            if next_move:
                if game.is_end():
                    result = game.board().result()
                    if result == '1-0':
                        gui.print_msg("%s wins." % game.root().headers["White"])
                    elif result == '0-1':
                        gui.print_msg("%s wins." % game.root().headers["Black"])
                    else:
                        gui.print_msg("Draw.")
                else:
                    game = game.variation(0)
            else:
                if game.parent is not None:
                    game = game.parent
                else:
                    gui.print_msg("This is the start of the game")

def main():
    if len(sys.argv) < 2:
        print "Usage: python view_pgn.py <filename>. File must be located in guerilla/games_played"
        sys.exit(0)
    view_game(sys.argv[1])

if __name__ == '__main__':
    main()
