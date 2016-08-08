import sys
sys.path.insert(0, '../pickles')

import subprocess
import time
import psutil
import os
import numpy as np
import math
import chess_game_parser as cgp
from hyper_parameters import *

# Modification from notbanker's stockfish.py https://gist.github.com/notbanker/3af51fd2d11ddcad7f16

def _recommended_threads():
    return psutil.cpu_count()-2

def _recommended_memory():
    return psutil.virtual_memory().available/(2*1024*1024)

def _numbers_in_file(fn):
    f = open( fn )
    return [ int(x) for x in f.readlines() ]


def stockfish( fen, seconds=1, threads=None, memory=None ):
    """ Return the evaluation for a position """
    return stockfish_scores( fen=fen, seconds=seconds, threads=threads, memory=memory)[-1]

def is_positional( scores ):
    """ True if the position is positional in nature """
    score_diff      = [ abs(sd) for sd in np.diff( scores ) ]
    return max( score_diff ) < 25

def stockfish_scores(fens, seconds=2, threads=None, memory=None, all_scores=False):
    """ 
        Uses stockfishes engine to evaluate a score for each board.
        Then uses a sigmoid to map the scores to a winning probability between 
        0 and 1 (see sigmoid_array for how the sigmoid was chosen)

            Inputs:
                boards[list of strings]:
                    list of board fens

            Outputs:
                values[list of floats]:
                    a list of values for each board ranging between 0 and 1
    """


    # Defaults
    memory = memory or _recommended_memory()
    threads = threads or _recommended_threads()
    binary = 'linux'

    # Shell out to Stockfish
    scores = []
    for fen in fens:

        cmd =  ' '.join( ['./helpers/stockfish_eval.sh' ,fen, str(seconds), binary, str(threads), str(memory) ] )
        # print cmd
        # try:
        output = subprocess.check_output(cmd, shell=True).strip().split('\n')
        # except subprocess.CalledProcessError e:

        # @TODO - fix hack. empties are returned by a segfault. It segfaults on this 8/4pp2/5kp1/5b1p/4K2P/3R2P1/5P2/8 (as an example, if segfaults on more)
        if output[0] == '':
            print "Warning: stockfish returned nothing. Skipping fen. Command was:\n%s" % (cmd)
            continue
        if len(output) == 2:
            score = 10000 if int(output[1]) > 0 else -10000
        else:
            score float(output[0])
        scores.append(score)

    return sigmoid_array(np.array(scores))


def sigmoid_array(values):
    ''' From: http://chesscomputer.tumblr.com/post/98632536555/using-the-stockfish-position-evaluation-score-to
        1000 cp lead almost guarantees a win (a sigmoid within that). From the looking at the graph to gather a few data point
        and using a sigmoid curve fitter an inaccurate function of 1/(1+e^(-0.00547x)) was decided on (by me, deal with it)
        Ideally this fitter function is learned, but this is just for testing so...'''
    return 1./(1. + np.exp(-0.00547*values))

print "Finished retrieving %d fens.\nBegin retrieving stockfish values.\n" % (len(fens))

def load_stockfish_values(filename='true_values.p'):
    ''' 
        Load stockfish values from a pickle file
        Inputs:
            filename[string]:
                pickle file to load values from
    '''
    stockfish_values = pickle.load(open(filename, 'rb'))
    return stockfish_values

def main():

    fens = cgp.load_fens()

    sf_scores = stockfish_scores(fens)
    # save stockfish_values
    pickle.dump(sf_scores, open('sf_scores.p', 'wb'))

if __name__ == "__main__":
    main()