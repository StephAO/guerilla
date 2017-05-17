# Benchmarking tests
import time
import chess

import guerilla.data_handler as dh
from guerilla.players import Guerilla


def complimentmax_search_bench(max_depth=3, num_rep=3, verbose=True):
    """
    Times how long searches of different depths takes.
    Input:
        max_depth [Int]
            Each search depth up to max_depth will be timed individually. (Inclusive)
        num_rep [Int]
            The number of times each search depth timing is repeated.
        verbose [Boolean]
            Whether results are printed in addition to being returned.
    Output:
        Result [Dict]
            Dictionary of Results.
    """

    output = {}

    # Random seed
    rnd_seed = 1234

    # Random board
    board = chess.Board('3r2k1/1br1qpbp/pp2p1p1/2pp3n/P2P1P2/1PP1P1P1/R2N2BP/1NR1Q1K1 w - - 5 24')

    if verbose:
        print "Each timing is the average of %d runs" % num_rep

    for i in range(1, max_depth + 1):
        # Create Guerilla with Random weights:
        with Guerilla('curious_george', verbose=False, seed=rnd_seed, search_params={'max_depth': i}) as g:

            # Time multiple repetitions
            avg_time = 0.0
            for _ in range(num_rep):
                start_time = time.time()
                g.get_move(board)
                avg_time += (time.time() - start_time) / num_rep

            key = 'Depth %d (s)' % i
            output[key] = avg_time

            if verbose:
                print '%s: %f' % (key, avg_time)

    return output


def search_types_bench(max_depth=3, time_limit=50, num_rep=1, verbose=True):
    """
    Times how long searches of different depths takes.
    Input:
        max_depth [Int]
            Maximum depth searched by Complimentmax.
        time_limit [Float]
            Time limit for RankPrune and IterativeDeepening.
        num_rep [Int]
            The number of times each search depth test is repeated.
        verbose [Boolean]
            Whether results are printed in addition to being returned.
    Output:
        Result [Dict]
            Dictionary of Results.
    """
    # Random seed
    rnd_seed = 123456

    # Random board
    board = chess.Board('3r2k1/1br1qpbp/pp2p1p1/2pp3n/P2P1P2/1PP1P1P1/R2N2BP/1NR1Q1K1 w - - 5 24')

    # Create Guerilla with Random weights:

    for st in ['iterativedeepening', 'minimax', 'rankprune']:
        sp = {'max_depth': max_depth} if st == 'minimax' else {'time_limit': time_limit}
        num_visits = None
        time_taken = num_evals = cache_hits = depth_reached = 0
        for _ in range(num_rep):
            with Guerilla('curious_george', search_type=st, seed=rnd_seed, verbose=False, search_params=sp) as g:
                # Time multiple repetitions
                start_time = time.time()
                g.get_move(board)
                time_taken += (time.time() - start_time) / num_rep

                # Increase output values
                if num_visits is None:
                    num_visits = g.search.num_visits
                else:
                    num_visits = [num_visits[i] + g.search.num_visits[i] for i in range(len(g.search.num_visits))]
                num_evals += g.search.num_evals
                cache_hits += g.search.cache_hits
                depth_reached += g.search.depth_reached

        print "Search type: %s, Average of %d repetition(s).\nTime Taken: %f\nNumber nodes visited by depth: %s \n" \
              "number of nodes evaluated: %d, cache hits: %d, depth reached: %d\n" % \
              (st, num_rep, time_taken, str([num_visits[i] / num_rep for i in range(len(num_visits))]),
               num_evals / num_rep, cache_hits / num_rep, depth_reached / num_rep)


def data_processing_bench():
    fen = '3r2k1/1br1qpbp/pp2p1p1/2pp3n/P2P1P2/1PP1P1P1/R2N2BP/1NR1Q1K1 w - - 5 24'
    input_types = ['bitmap', 'giraffe', 'movemap']
    for input_type in input_types:
        start_time = time.time()
        for _ in xrange(1000):
            dh.fen_to_nn_input(fen, input_type)
        print '1000 iterations of fen to %s took %f seconds' % \
              (input_type, time.time() - start_time)


def nn_evaluation_bench():
    fen = '3r2k1/1br1qpbp/pp2p1p1/2pp3n/P2P1P2/1PP1P1P1/R2N2BP/1NR1Q1K1 w - - 5 24'
    input_types = ['bitmap', 'giraffe', 'movemap']
    for input_type in input_types:
        for num_fc in xrange(1, 5):
            start_time = time.time()
            with Guerilla('curious_george', verbose=False,
                          nn_params={'NUM_FC': num_fc, 'NN_INPUT_TYPE': input_type}) as g:
                for _ in xrange(100):
                    g.nn.evaluate(fen)
            print '100 iterations of evaluate using %s with %d fc layers took %f seconds' % \
                  (input_type, num_fc, time.time() - start_time)


def run_benchmark_tests():
    benchmarks = {
        'Complimentmax Search': complimentmax_search_bench,
        # 'Search Types': search_types_bench,
        # 'Data Processing': data_processing_bench,
        # 'Evaluation': nn_evaluation_bench
    }

    print "\nRunning Benchmarks..."
    for name, test in benchmarks.iteritems():
        print "\nRunning " + name + " Benchmark..."
        test()


if __name__ == '__main__':
    run_benchmark_tests()
