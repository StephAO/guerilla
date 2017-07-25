import pandas as pd
import numpy as np
import chess
import matplotlib.pyplot as plt

import guerilla.train.stockfish_eval as sf
import guerilla.train.chess_game_parser as cgp
import guerilla.data_handler as dh
from guerilla.players import Guerilla


def metric_by_move(weight_files, labels=None, num_values=100, metric='mean', verbose=False):
    # Displays accuracy metric by move
    # Potential metrics are 'err_mean', 'err_variance', 'mean', 'variance'
    # Respectively: mean error (abs(predicted - actual)), variance of mean, mean score, variance of score

    labels = [weight_files] if labels is None else labels

    # Load data
    if verbose:
        print "Loading data.."
    actual = sf.load_stockfish_values(num_values=num_values)
    fens = cgp.load_fens(num_values=len(actual))
    if verbose:
        print "Loaded %d FENs and scores." % len(fens)

    # Predict values and get move numbers
    for w, weight_file in enumerate(weight_files):
        if verbose:
            print "Generating predictions..."
        move_num = [0] * len(fens)  # halfmove
        with Guerilla('Harambe', load_file=weight_file) as g:
            predicted = get_predictions(g, fens, verbose)

            for i, fen in enumerate(fens):
                move_num[i] = 2 * (int(dh.strip_fen(fen, 5)) - 1) + (1 if dh.black_is_next(fen) else 0)

        # Convert to dataframe for plotting
        if verbose:
            print "Converting to dataframe..."
        df = pd.DataFrame({'fens': fens, 'move_num': move_num, 'actual': actual, 'predicted': predicted})
        df['abs_error'] = np.abs(df['actual'] - df['predicted'])

        # Group
        if verbose:
            print "Grouping..."
        g = df.groupby('move_num')
        mean_data = g.aggregate(np.mean)
        var_data = g.aggregate(np.var)

        if metric == 'err_mean':
            x = mean_data['abs_error']
        elif metric == 'err_variance':
            x = var_data['abs_error']
        elif metric == 'mean':
            x = mean_data['predicted']
        elif metric == 'variance':
            x = var_data['predicted']
        else:
            raise ValueError("Metric %s has not been implemented!" % metric)

        plt.plot(x, label=labels[w], color=plt.cm.cool(w * 1.0 / len(weight_files)))

    if metric == 'mean':
        plt.plot(mean_data['actual'], label='actual', color='k')
    elif metric == 'variance':
        plt.plot(var_data['actual'], label='actual', color='k')
    plt.xlabel('Half-Move')
    plt.ylabel('%s' % metric)
    plt.xlim([0, 100])
    # plt.ylim([0, int(7e6)])
    plt.title('%s by Move' % metric)
    plt.legend()
    plt.show()


def prediction_distribution(weight_files, labels=None, bins=None, num_values=500, verbose=False):
    actual = sf.load_stockfish_values(num_values=num_values)
    fens = cgp.load_fens(num_values=num_values)

    labels = [weight_files] if labels is None else labels

    # Predict values and get move numbers
    for w, weight_file in enumerate(weight_files):
        if verbose:
            print "Generating predictions for %s..." % weight_file
        with Guerilla('Harambe', load_file=weight_file) as g:
            predicted = get_predictions(g, fens, verbose)

        plt.hist(predicted, bins=bins, linewidth=1.5, alpha=1.0, label=labels[w], histtype='step',
                 color=plt.cm.cool(w * 1.0 / len(weight_files)))
        plt.title(weight_file)
        # plt.hist()

    plt.hist(actual, bins=bins, linewidth=1.5, label='SF', histtype='step')

    plt.legend()
    plt.title('Score Histogram')
    plt.show()


def error_by_depth(weight_file, min_depth=1, max_depth=3, num_values=1000):
    actual = sf.load_stockfish_values(num_values=num_values)
    fens = cgp.load_fens(num_values=num_values)

    # Parameters
    binwidth = 25

    with Guerilla('Harambe', load_file=weight_file, search_type='minimax') as g:
        for depth in range(min_depth, max_depth + 1):
            # Set depth
            g.search.max_depth = depth
            g.search.clear_cache()

            predicted = get_predictions(g, fens, mode='search', verbose=True)  # Get predictions

            error = abs(np.array(actual) - np.array(predicted))

            plt.subplot((max_depth - min_depth + 1), 1, depth)

            # Make sum = 1
            weights = np.ones_like(error) / float(len(error))
            plt.hist(error, weights=weights, bins=range(0, 5000 + binwidth, binwidth))

            # Properties
            plt.ylim([0, 1.0])
            plt.title('Depth %s' % depth)
            plt.axvline(x=np.mean(error), color='k')
            print "Depth %d MEAN: %f STD: %f VAR: %f" % (depth, np.mean(error), np.std(error), np.var(error))

            err_greater = [x for x in error if x > 500]
            err_lesser = [x for x in error if x <= 500]
            print "Depth %d: %d predictions with an error > 500" % (depth, len(err_greater))
            print "Depth %d: Mean of errors > 500 is %f" % (depth, np.mean(err_greater))
            print "Depth %d: Mean of errors < 500 is %f" % (depth, np.mean(err_lesser))

    plt.ylabel('Frequency')
    plt.xlabel('Abs Error')
    plt.show()


def distr_by_depth(weight_file, fen, min_depth=1, max_depth=3):
    actual = sf.stockfish_eval_fn(fen)
    print "Actual value %f" % actual

    # Parameters
    binwidth = 25

    root_score = {}
    scores_by_depth = {depth: [] for depth in range(min_depth, max_depth + 1)}
    with Guerilla('Harambe', load_file=weight_file, search_type='iterativedeepening') as g:
        for depth in range(min_depth, max_depth + 1):
            # Search
            print "Running depth %d" % depth
            g.search.max_depth = depth
            g.search.ab_prune = False  # turn off pruning
            board = chess.Board(fen)
            score, move, _ = g.search.run(board)
            print "%f %s" % (score, move)
            root_score[depth] = score

            # Travel through depths
            queue = [g.search.root]

            while queue != []:
                curr = queue.pop(0)  # pop front of queue
                # Push children to queue
                for child in curr.children.itervalues():
                    queue.append(child)

                # only store leaf boards
                if curr.depth == depth:
                    scores_by_depth[curr.depth].append(curr.value)

    # Plot
    for depth, values in scores_by_depth.iteritems():
        plt.subplot((max_depth - min_depth + 1), 1, depth)

        # Make sum = 1
        weights = np.ones_like(values) / float(len(values))
        plt.hist(values, weights=weights, bins=range(-5000, 5000 + binwidth, binwidth))

        # Properties
        plt.ylim([0, 1.0])
        plt.title('Depth %s' % depth)
        plt.axvline(x=np.mean(actual), color='k')
        plt.axvline(x=root_score[depth], color='r')

    plt.ylabel('Frequency')
    plt.xlabel('Value')
    plt.show()


def get_predictions(guerilla, fens, mode=None, verbose=False):
    mode = 'eval' if mode is None else mode

    if verbose:
        print "Generating predictions for %s..." % guerilla.name
    predictions = [0] * len(fens)
    for i, fen in enumerate(fens):
        if mode == 'eval':
            predictions[i] = guerilla.get_cp_adv_white(fen)
        elif mode == 'search':
            board = chess.Board(fen)
            score, _, _ = guerilla.search.run(board)
            predictions[i] = score
        else:
            raise ValueError("Prediction mode %s has not been implemented!" % mode)

        if verbose:
            print_perc = 5
            if (i % (len(fens) / (100.0 / print_perc)) - 100.0 / len(fens)) < 0:
                print "%d%% " % (i / (len(fens) / 100.0)),

    print ''
    return predictions

def main():
    # labels=[str(i) for i in range(25, 125, 25)]
    # weight_files = ['var_check_2_%s.p' % label for label in labels]
    # weight_files = ['var_check_old/var_check_2_250.p', 'var_check_old/var_check_2_100.p', ]
    # labels = ['TD', 'TD + loss']
    # metric_by_move(weight_files, labels, num_values=5000, verbose=True, metric='variance')
    # bins = range(-5000, 5100, 100)
    #
    # # Add original
    # weight_files = ['6811.p'] + weight_files
    # labels = ['no_TD'] + labels
    #
    # prediction_distribution(weight_files, labels=labels, bins=bins, num_values=10000, verbose=True)

    # error_by_depth('6811.p', num_values=2000)
    distr_by_depth('6811.p', fen='1r3rk1/8/3p3p/p1qP2p1/R1b1P3/2Np1P2/1P1Q1RP1/6K1 w - - 0 1')

if __name__ == '__main__':
    main()
