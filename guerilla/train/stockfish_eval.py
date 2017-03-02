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
from pkg_resources import resource_filename
import guerilla.train.chess_game_parser as cgp


# Modification from notbanker's stockfish.py https://gist.github.com/notbanker/3af51fd2d11ddcad7f16

def stockfish_scores(generate_time, seconds=1, threads=None, memory=None, all_scores=False, num_attempt = 3):
    """ 
        Uses stockfishes engine to evaluate a score for each board.
        Then uses a sigmoid to map the scores to a winning probability between 
        0 and 1 (see sigmoid_array for how the sigmoid was chosen).

            Inputs:
                boards[list of strings]:
                    list of board fens
                num_attempt [Int]
                    The number of times to attempt to score a fen.

            Outputs:
                values[list of floats]:
                    a list of values for each board ranging between 0 and 1
    """

    sf_num = 0
    if os.path.isfile(resource_filename('guerilla', 'data/extracted_data/sf_num.txt')):
        with open(resource_filename('guerilla', 'data/extracted_data/sf_num.txt'), 'r') as f:
            l = f.readline()
            sf_num = int(l)

    batch_size = 5

    with open(resource_filename('guerilla', 'data/extracted_data/fens.nsv'), 'r') as fen_file:
        with open(resource_filename('guerilla', 'data/extracted_data/sf_values.nsv'), 'a') as sf_file:

            for i in xrange(sf_num):
                fen_file.readline()

            # Shell out to Stockfish
            scores = []
            start_time = time.time()
            while (time.time() - start_time) < generate_time:
                fen = fen_file.readline().strip()
                print fen

                if fen == "":
                    break

                score = get_stockfish_score(fen, seconds = seconds, threads = threads, memory = memory, num_attempt=num_attempt)

                if score is None:
                    print "Failed to score fen '%s' after %d attempts. Exiting." % (fen, num_attempt)
                    break

                scores.append(score)
                sf_num += 1

                if (sf_num + 1) % batch_size == 0:
                    mapped_scores = sigmoid_array(np.array(scores))  # type: np.ndarray
                    for mscore in mapped_scores:
                        sf_file.write(str(mscore) + '\n')
                    scores = []

                    with open(resource_filename('guerilla', 'data/extracted_data/sf_num.txt'), 'w') as num_file:
                        num_file.write(str(sf_num))

            mapped_scores = sigmoid_array(np.array(scores))  # type: np.ndarray
            for mscore in mapped_scores:
                sf_file.write(str(mscore) + '\n')

    # Write out the index of the next fen to score
    with open(resource_filename('guerilla', 'data/extracted_data/sf_num.txt'), 'w') as num_file:
        num_file.write(str(sf_num))


def get_stockfish_score(fen, seconds, threads=None, memory=None, num_attempt=1, max_depth=None):
    """
    Input:
        fen [String]
            Chess board to evaluate.
        seconds [Int]
            Number of seconds to evaluate the board.
        threads [Int]
            Number of threads to use for stockfish.
        memory [Int]
            Amount of memory to use for stockfish
        num_attempt [Int]
            Number of attempts which should be made to get a stockfish score for the given fen.
        max_depth [Int]
            Max depth to search to.
    Output:
        score [Float]
            Stockfish score. Returns None if no score found.
    """

    # Base for mate scoring
    MATE_BASE = 5000 # 50 pawn advantage (>5 queens)

    memory = memory or psutil.virtual_memory().available / (2 * 1024 * 1024)
    threads = threads or psutil.cpu_count() - 2
    binary = 'linux'

    cmd = ' '.join([(resource_filename('guerilla.train', '/stockfish_eval.sh')), fen, str(seconds), binary,
                    str(threads), str(memory), str(max_depth) if max_depth else ''])

    attempt = 0
    output = None
    while attempt < num_attempt:
        try:
            output = subprocess.check_output(cmd, shell=True).strip().split('\n')
            if output is not None:
                break
        except subprocess.CalledProcessError as e:
            print e

        attempt += 1

    if output is None:
        return output
    elif output[0] == '':
        print "Warning: stockfish returned nothing. Command was:\n%s" % cmd
        return None

    if len(output) == 2:
        print "ERROR: Too long (len > 1) stockfish output. Command was:\n%s" % cmd
        return None

    output = output[0].split(' ')
    if output[0] == 'mate':
        mate_in = int(output[1])
        if mate_in == 0:
            # avoids division by zero
            mate_in = 1e-9
        score = MATE_BASE * (1 + 1.0 / abs(mate_in))
        if mate_in < 0:
            # White will LOSE in mate_in turns, therefore make the score negative
            score *= -1
    else:  # cp
        score = float(output[1])

    return score

def sigmoid_array(values):
    """ From: http://chesscomputer.tumblr.com/post/98632536555/using-the-stockfish-position-evaluation-score-to
        1000 cp lead almost guarantees a win (a sigmoid within that). From the looking at the graph to gather a
        few data points and using a sigmoid curve fitter an inaccurate function of 1/(1+e^(-0.00547x)) was decided
        on (by me, deal with it).
        Ideally this fitter function is learned, However using this function directly
        on STS did result in a high results."""
    return 1. / (1. + np.exp(-0.00547 * values))


def load_stockfish_values(filename='sf_values.nsv', num_values=None):
    """
        Load stockfish values from a file
        Inputs:
            filename[string]:
                file to load values from
            num_values[int]:
                Max number of stockfish values to return. 
                (will return min of num_values and number of values stored in file)
        Outpus:
            stockfish_values[list of floats]:
                list of stockfish_values corresponding to the order of
                fens in fens.nsv
    """
    stockfish_values = []
    count = 0
    with open(resource_filename('guerilla', 'data/extracted_data/' + filename), 'r') as sf_file:
        for line in sf_file:
            stockfish_values.append(float(line.strip()))
            count += 1
            if num_values is not None and count >= num_values:
                break

    return stockfish_values


def stockfish_eval_fn(fen, seconds=0.3, max_depth=1, num_attempt=3):
    """
    Evaluates the given fen using stockfish. Returns P(win).
    Input:
        fen [String]
            FEN to evaluate.
    Output:
        value [Float]
            P(win) of input FEN.
    """

    raw_value = get_stockfish_score(fen, seconds=seconds, max_depth=max_depth, num_attempt=num_attempt)

    return sigmoid_array(raw_value)

def main():

    generate_time = int(raw_input("How many seconds do you want to generate stockfish values for?: "))

    print "Evaluating fens for %d seconds, spending 1 second on each" % (generate_time)

    stockfish_scores(generate_time)

if __name__ == "__main__":

    main()
