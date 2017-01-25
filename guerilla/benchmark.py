# Benchmarking tests
import time
import chess

from guerilla.players import Guerilla


def search_bench(max_depth=3, num_rep=3, verbose=True):
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

    # Random board
    board = chess.Board('3r2k1/1br1qpbp/pp2p1p1/2pp3n/P2P1P2/1PP1P1P1/R2N2BP/1NR1Q1K1 w - - 5 24')

    if verbose:
        print "Each timing is the average of %d runs" % num_rep

    # Create Guerilla with Random weights:
    with Guerilla('curious_george','w', verbose=False) as g:
        for i in range(3, max_depth + 1):
            g.search.max_depth = i

            # Time multiple repetitions
            avg_time = 0.0
            for _ in range(num_rep):
                start_time = time.time()
                g.get_move(board)
                avg_time += (time.time() - start_time)/num_rep

            key = 'Depth %d (s)' % i
            output[key] = avg_time

            if verbose:
                print '%s: %f' % (key, avg_time)

    return output

def run_benchmark_tests():
    benchmarks = {
        'Search': search_bench
    }

    print "\nRunning Benchmarks...\n"
    for name, test in benchmarks.iteritems():
        print "Running " + name + " Benchmark..."
        test()

if __name__ == '__main__':
    run_benchmark_tests()