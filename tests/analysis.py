import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import guerilla.train.stockfish_eval as sf
import guerilla.train.chess_game_parser as cgp
import guerilla.data_handler as dh
from guerilla.players import Guerilla


def accuracy_by_move(weight_file, sf_file=None, fens_file=None, verbose=False):
    # Predicts the accuracy by move

    # Load data
    if verbose:
        print "Loading data.."
    actual = sf.load_stockfish_values(num_values=10000)
    fens = cgp.load_fens(num_values=len(actual))
    if verbose:
        print "Loaded %d FENs and scores." % len(fens)

    # Predict values and get move numbers
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

    mean_data.plot(y='abs_error')
    plt.xlabel('Half-Move')
    plt.ylabel('Absolute error (Mean)')
    plt.xlim([0, 100])
    plt.ylim([0, 500])
    plt.show()


def main():
    accuracy_by_move('5083.p', verbose=True)


if __name__ == '__main__':
    main()
