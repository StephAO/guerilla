"""
Partial UCI protocol implementation for Guerilla.
UCI Info: http://wbec-ridderkerk.nl/html/UCIProtocol.html

Usage:

    uci
	    tell engine to use the uci (universal chess interface),

    debug [ on | off ]
	    switch the debug mode of the engine on and off.

    isready
	    this is used to synchronize the engine with the GUI. When the GUI has sent a command or
	    multiple commands that can take some time to complete,
	    this command can be used to wait for the engine to be ready again or
	    to ping the engine to find out if it is still alive.

    setoption name  [value ]
        this is sent to the engine when the user wants to change the internal parameters

    ucinewgame
        this is sent to the engine when the next search (started with "position" and "go") will be from
        a different game. This can be a new game the engine should play or a new game it should analyse but
        also the next position from a testsuite with positions only.
   
    position [fen  | startpos ]  moves  .... 
        set up the position described in fenstring on the internal board and play the moves on the internal chess board.

    go
        start calculating on the current position set up with the "position" command.
        There are a number of commands that can follow this command, all will be sent in the same string.
        If one command is not send its value should be interpreted as it would not influence the search.
            depth 
                search x plies only.
            movetime 
                search exactly x mseconds
    quit
	    quit the program as soon as possible
"""

import chess
import sys
from guerilla.players import Guerilla

# TODO: Speed up booting by waiting for 'setoptions' or 'isready' to setup internal parameters

NAME = 'Guerilla 0.1.0'
AUTHOR = 'M. Aroca-Ouellette, S. Aroca-Ouellette'
OPTIONS = []


class UCI:
    def __init__(self, guerilla):
        self.guerilla = guerilla
        self.curr_board = None

    def process_command(self, guerilla, commands):
        """
        Processes the input commands. Ignores unknown commands.
        Input [List of Strings]
            List of string commands.
        Output:
            True if the command is to quit the program. False otherwise.
        """

        cmd = commands[0]
        modifiers = commands[1:]

        if cmd == 'uci':
            output = 'id name %s\nid author %s\n' % (NAME, AUTHOR)

            if OPTIONS:
                for option in OPTIONS:
                    output += option + '\n'

            print output + 'uciok'
        elif cmd == 'debug':
            # Toggle verbose
            if modifiers[0] == 'on':
                self.set_verbose(True)
            elif modifiers[0] == 'off':
                self.set_verbose(False)
        elif cmd == 'isready':
            # Always ready since not using multiple threads.
            print 'readyok'
        elif cmd == 'setoption':
            pass
        elif cmd == 'ucinewgame':
            self.curr_board = chess.Board()
        elif cmd == 'position':
            moves = None
            if modifiers[0] == 'fen':
                # Next 6 items are fen
                fen = ' '.join(modifiers[1:7])
                try:
                    self.curr_board = chess.Board(fen)
                    moves = modifiers[7:]
                except ValueError:
                    # Invalid fen
                    pass
            elif modifiers[0] == 'startpos':
                self.curr_board = chess.Board()  # starting positions
                moves = modifiers[1:]

            # Apply moves
            if moves and moves[0] == 'moves':
                for move in moves[1:]:
                    self.curr_board.push_uci(move)
        elif cmd == 'go':
            if modifiers[0] == 'movetime':
                self.guerilla.search.max_depth = None
                self.guerilla.search.time_limit = float(modifiers[1]) / 1000  # since in milliseconds
            elif modifiers[0] == 'depth':
                self.guerilla.search.max_depth = int(modifiers[1])
                self.guerilla.search.time_limit = float("inf")

            print "bestmove %s" % str(self.guerilla.get_move(self.curr_board))
        elif cmd == 'quit':
            return True

        # flush buffer
        sys.stdout.flush()

        return False

    def set_verbose(self, verbose):
        """
        Sets the verbosity of the different components of Guerilla based on the 'verbose' input.
        Inputs:
            guerilla [Guerilla]
                Guerilla instance.
            verbose [Boolean]
                The value to set for verbosity of the various Guerilla components.
        """

        components = [self.guerilla.nn]

        for component in components:
            component.verbose = verbose


def main():
    with Guerilla('Harambe', load_file='default.p', search_type='iterativedeepening',
                  search_params={'time_limit': 1}) as g:
        uci = UCI(g)
        # Get commands
        while True:
            quit = uci.process_command(g, raw_input().split(' '))
            if quit:
                break


if __name__ == '__main__':
    main()
