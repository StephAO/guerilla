import pandas as pd
import numpy as np
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
        predicted = [0] * len(fens)
        move_num = [0] * len(fens)  # halfmove
        with Guerilla('Harambe', load_file=weight_file) as g:
            for i, fen in enumerate(fens):
                predicted[i] = g.get_cp_adv_white(fen)
                move_num[i] = 2 * (int(dh.strip_fen(fen, 5)) - 1) + (1 if dh.black_is_next(fen) else 0)

                if verbose:
                    print_perc = 5  # TODO: Refactor
                    if (i % (len(fens) / (100.0 / print_perc)) - 100.0 / len(fens)) < 0:
                        print "%d%% " % (i / (len(fens) / 100.0)),

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
        predicted = [0] * len(fens)
        with Guerilla('Harambe', load_file=weight_file) as g:
            for i, fen in enumerate(fens):
                predicted[i] = g.get_cp_adv_white(fen)

                if verbose:
                    print_perc = 5  # TODO: Refactor
                    if (i % (len(fens) / (100.0 / print_perc)) - 100.0 / len(fens)) < 0:
                        print "%d%% " % (i / (len(fens) / 100.0)),

        plt.hist(predicted, bins=bins, linewidth=1.5, alpha=1.0, label=labels[w], histtype='step',
                 color=plt.cm.cool(w * 1.0 / len(weight_files)))
        plt.title(weight_file)
        # plt.hist()

    plt.hist(actual, bins=bins, linewidth=1.5, label='SF', histtype='step')

    plt.legend()
    plt.title('Score Histogram')
    plt.show()

def main():
    # labels=[str(i) for i in range(25, 125, 25)]
    # weight_files = ['var_check_2_%s.p' % label for label in labels]
    weight_files = ['var_check_old/var_check_2_250.p', 'var_check_old/var_check_2_100.p', ]
    labels = ['TD', 'TD + loss']
    metric_by_move(weight_files, labels, num_values=5000, verbose=True, metric='variance')
    bins = range(-5000, 5100, 100)

    # Add original
    weight_files = ['6811.p'] + weight_files
    labels = ['no_TD'] + labels

    prediction_distribution(weight_files, labels=labels, bins=bins, num_values=10000, verbose=True)

if __name__ == '__main__':
    main()
