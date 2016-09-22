"""
Uses Stockfish to score board states in the range (0,1).
"""

import subprocess
import time
import psutil
import os
import numpy as np
import math
import pickle
import chess_game_parser as cgp


# Modification from notbanker's stockfish.py https://gist.github.com/notbanker/3af51fd2d11ddcad7f16

def stockfish_scores(fens, seconds=1, threads=None, memory=None, all_scores=False):
    """ 
        Uses stockfishes engine to evaluate a score for each board.
        Then uses a sigmoid to map the scores to a winning probability between 
        0 and 1 (see sigmoid_array for how the sigmoid was chosen).

            Inputs:
                boards[list of strings]:
                    list of board fens

            Outputs:
                values[list of floats]:
                    a list of values for each board ranging between 0 and 1
    """

    # Defaults
    memory = memory or psutil.virtual_memory().available / (2 * 1024 * 1024)
    threads = threads or psutil.cpu_count() - 2
    binary = 'linux'

    # Shell out to Stockfish
    scores = []
    percent_done = 0
    num_fens = len(fens)
    for i, fen in enumerate(fens):
        if math.floor(i * 100 / num_fens) > percent_done:
            percent_done = math.floor(i * 100 / num_fens)
            print '|' + '#' * int(percent_done) + " %d " % (percent_done) + "%" + " done"
        cmd = ' '.join([(dir_path + '/stockfish_eval.sh'), fen, str(seconds), binary, str(threads), str(memory)])
        # print cmd
        # try:
        output = subprocess.check_output(cmd, shell=True).strip().split('\n')
        # except subprocess.CalledProcessError e:

        if output[0] == '':
            print "Warning: stockfish returned nothing. Skipping fen. Command was:\n%s" % cmd
            continue
        if len(output) == 2:
            score = 100000. if int(output[1]) > 0 else -100000.
        else:
            score = float(output[0])
        scores.append(score)

    return sigmoid_array(np.array(scores))


def sigmoid_array(values):
    """ From: http://chesscomputer.tumblr.com/post/98632536555/using-the-stockfish-position-evaluation-score-to
        1000 cp lead almost guarantees a win (a sigmoid within that). From the looking at the graph to gather a
        few data points and using a sigmoid curve fitter an inaccurate function of 1/(1+e^(-0.00547x)) was decided
        on (by me, deal with it).
        Ideally this fitter function is learned, but this is just for testing so..."""
    # TODO S: Improve sigmoid mapping.

    return 1. / (1. + np.exp(-0.00547 * values))


def load_stockfish_values(filename='sf_scores.p'):
    """
        Load stockfish values from a pickle file
        Inputs:
            filename[string]:
                pickle file to load values from
    """
    full_path = dir_path + "/../pickles/" + filename
    stockfish_values = pickle.load(open(full_path, 'rb'))
    return stockfish_values


def main():
    fens = cgp.load_fens()

    print "Evaluating %d fens for 1 seconds each" % (len(fens))

    sf_scores = stockfish_scores(fens)

    # save stockfish_values
    pickle_path = dir_path + '/../pickles/sf_scores.p'
    pickle.dump(sf_scores, open(pickle_path, 'wb'))

dir_path = dir_path = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    main()
