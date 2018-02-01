"""
Uses Stockfish to provide centipawn scores.
"""

import subprocess
import time
import psutil
import os
import numpy as np
import guerilla.data_handler as dh
from pkg_resources import resource_filename


# Modification from notbanker's stockfish.py https://gist.github.com/notbanker/3af51fd2d11ddcad7f16

def stockfish_scores(generate_time, seconds=1.0, threads=None, memory=None, num_score=None, num_attempt=3):
    """ 
        Uses stockfishes engine to evaluate a score for each board.

            Inputs:
                boards[list of strings]:
                    list of board fens
                num_attempt [Int]
                    The number of times to attempt to score a fen.

            Outputs:
                values[list of floats]:
                    a list of values for each board ranging between 0 and 1
    """

    print "Evaluating fens for {} seconds, spending {} second(s) on each".format(generate_time, seconds)

    sf_num = 0
    if os.path.isfile(resource_filename('guerilla', 'data/extracted_data/sf_num.txt')):
        with open(resource_filename('guerilla', 'data/extracted_data/sf_num.txt'), 'r') as f:
            l = f.readline()
            sf_num = int(l)

    print "Starting from FEN #{}...".format(sf_num)

    batch_size = 5

    with open(resource_filename('guerilla', 'data/extracted_data/fens.csv'), 'r') as fen_file:
        with open(resource_filename('guerilla', 'data/extracted_data/cp_values.csv'), 'a') as sf_file:

            for i in xrange(sf_num):
                fen_file.readline()

            # Shell out to Stockfish
            scores = []
            start_time = time.time()
            while (time.time() - start_time) < generate_time and (num_score is None or sf_num < num_score):
                fen = fen_file.readline().strip()
                print fen
                assert (dh.white_is_next(fen))

                if fen == "":
                    break

                score = get_stockfish_score(fen, seconds=seconds, threads=threads, memory=memory,
                                            num_attempt=num_attempt)

                if score is None:
                    print "Failed to score fen '%s' after %d attempts. Exiting." % (fen, num_attempt)
                    break

                scores.append(score)
                sf_num += 1

                if (sf_num + 1) % batch_size == 0:
                    for mscore in scores:
                        sf_file.write(str(mscore) + '\n')
                    scores = []

                    with open(resource_filename('guerilla', 'data/extracted_data/sf_num.txt'), 'w') as num_file:
                        num_file.write(str(sf_num))

            for mscore in scores:
                sf_file.write(str(mscore) + '\n')

    # Write out the index of the next fen to score
    with open(resource_filename('guerilla', 'data/extracted_data/sf_num.txt'), 'w') as num_file:
        num_file.write(str(sf_num))


def get_stockfish_score(fen, seconds, threads=None, memory=None, num_attempt=1, max_depth=None,
                        sleep_length=1, attempt_step=5, attempt_step_inc=0.5):
    """
    Input:
        fen [String]
            Chess board to evaluate.
        seconds [Int]
            Initial number of seconds to evaluate the board.
        threads [Int]
            Number of threads to use for stockfish.
        memory [Int]
            Amount of memory to use for stockfish
        num_attempt [Int]
            Number of attempts which should be made to get a stockfish score for the given fen.
        max_depth [Int]
            Max depth to search to.
        sleep_length [Int]
            Amount of time to sleep between attempts.
        attempt_step [Int]
            Number of attempts to make before increasing command time.
        attempt_step_inc [Float]
            Amount of time to increase command time by at each attempt_step.
    Output:
        score [Float]
            Stockfish score. Returns None if no score found.
    """

    memory = memory or psutil.virtual_memory().available / (2 * 1024 * 1024)
    threads = threads or psutil.cpu_count()
    binary = 'linux'

    attempt = 0
    output = None
    cmd = None
    while attempt < num_attempt:
        try:
            cmd = ' '.join([(resource_filename('guerilla.train', '/stockfish_eval.sh')), fen, str(seconds), binary,
                            str(threads), str(memory), str(max_depth) if max_depth else ''])

            output = subprocess.check_output(cmd, shell=True).strip().split('\n')
            if output is not None and output[0] != '':
                break
        except subprocess.CalledProcessError as e:
            print e

        attempt += 1
        time.sleep(sleep_length)
        if attempt % attempt_step == 0:
            seconds += attempt_step_inc
            print "Increased command time to {} seconds...".format(seconds)

    if output is None:
        return output
    elif output[0] == '':
        print "Warning: stockfish returned nothing ({}). Command was:\n{}".format(output, cmd)
        return None

    if len(output) == 2:
        print "ERROR: Too long (len > 1) stockfish output. Command was:\n{}".format(cmd)
        return None

    output = output[0].split(' ')
    if output[0] == 'mate':
        mate_in = int(output[1])
        if mate_in == 0:
            # MEANS WHITE LOST (i.e. white's turn, but white is checkmated)
            mate_in = -1

        # Checkmate in 'mate_in' moves
        score = dh.WIN_VALUE - (abs(mate_in) - 1) * 500
        if mate_in < 0:
            # White will LOSE in mate_in turns, therefore make the score negative
            score *= -1
    else:  # cp
        score = float(output[1])

    return score


def sigmoid_array(values):
    """ NO LONGER USED
        NOTE: Not in use since switched to linear output.
        From: http://chesscomputer.tumblr.com/post/98632536555/using-the-stockfish-position-evaluation-score-to
        1000 cp lead almost guarantees a win (a sigmoid within that). From the looking at the graph to gather a
        few data points and using a sigmoid curve fitter an inaccurate function of 1/(1+e^(-0.00547x)) was decided
        on (by me, deal with it).
        Ideally this fitter function is learned, However using this function directly
        on STS did result in a high results."""
    return 1. / (1. + np.exp(-0.00547 * values))


def logit(value):
    """
    NO LONGER USED
    Inverse of sigmoid array function. Converts from P(win) to centipawn advantage.
    """
    if value <= 0.0000000000015:
        return dh.LOSE_VALUE
    elif value >= 0.999999999999:
        return dh.WIN_VALUE
    else:
        return (1. / 0.00547) * (np.log(value) - np.log(1. - value))


def reverse_true_values_to_cp():
    """
    Converts sf_values file where scores are P(win) to cp_values file where scores are centipawn advantage. 
    """
    with open(resource_filename('guerilla', 'data/extracted_data/sf_values.csv'), 'r') as input_file, \
            open(resource_filename('guerilla', 'data/extracted_data/cp_values.csv'), 'w') as output_file:
        stime = time.time()
        for line in input_file:
            # print line
            pw = float(line)
            cp = int(logit(pw))
            output_file.write(str(cp) + "\n")
        print time.time() - stime


def load_stockfish_values(filename='cp_values.csv', num_values=None):
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
                fens in fens.csv
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
    Evaluates the given fen using stockfish. Returns centipawn advantage.
    Input:
        fen [String]
            FEN to evaluate.
    Output:
        value [Float]
            P(win) of input FEN.
    """

    raw_value = get_stockfish_score(fen, seconds=seconds, max_depth=max_depth, num_attempt=num_attempt)

    return raw_value


def main():
    generate_time = int(raw_input("How many seconds do you want to generate stockfish values for?: "))
    seconds = 0.5

    stockfish_scores(generate_time=generate_time, seconds=seconds, num_attempt=10)


if __name__ == "__main__":
    main()
