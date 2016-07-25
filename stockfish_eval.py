import subprocess
import time
import psutil
import os
import numpy as np

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

def stockfish_scores(fen, seconds=1, threads=None, memory=None, all_scores=False):
    """ Call stockfish engine and return vector of evaluation score """

    # Defaults
    memory = memory or _recommended_memory()
    threads = threads or _recommended_threads()
    binary = 'linux'

    # Shell out to Stockfish
    cmd =  ' '.join( ['./stockfish_eval.sh' ,fen, str(seconds) , binary, str(threads), str(memory) ] )
    output = subprocess.check_output(cmd, shell=True).strip().split('\n')
    
    if len(output) == 2:
        return 100 if int(output[1]) > 0 else -100
    else:
        return float(output[0])/100.0
