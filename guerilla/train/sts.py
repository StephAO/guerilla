#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Containts STS Evaluation functions

import time
import chess
import sys
from pkg_resources import resource_filename
from guerilla.players import *

sts_strat_files = ['activity_of_king', 'advancement_of_abc_pawns', 'advancement_of_fgh_pawns', 'bishop_vs_knight',
                   'center_control', 'knight_outposts', 'offer_of_simplification', 'open_files_and_diagonals',
                   'pawn_play_in_the_center', 'queens_and_rooks_to_the_7th_rank',
                   'recapturing', 'simplification', 'square_vacancy', 'undermining', 'avoid_pointless_exchange']

sts_piece_files = ['pawn', 'bishop', 'rook', 'knight', 'queen', 'king']


def eval_sts(player, mode="strategy", step_size=1, verbose=False, num_track=25, num_per_test=2):
    """
    Evaluates the given player using the strategic test suite. Returns a score and a maximum score.
        Inputs:
            player [Player]
                Player to be tested.
            mode [List of Strings] or [String]
                Selects the test mode(s), see below for options. By default runs "strategy".
                    "strategy": runs all strategic tests
                    "pieces" : runs all piece tests
                    other: specific EPD file
            step_size [int]:
                Number of positions to skip each eval. 
                If 1, run all positions. If 10, run every 10th position
        Outputs:
            scores [List of Integers]
                List of scores the player received on the each test mode. Same order as input.
            max_scores [Integer]
                List of highest possible scores on each test type. Same order as score output.
    """

    # Handle input
    if not isinstance(player, Player):
        raise ValueError("Invalid input! Player must derive abstract Player class.")

    if type(mode) is not list:
        mode = [mode]

    # vars
    scores = []
    max_scores = []
    count = 0

    # Best and worst arrays
    best = []
    worst = []
    num_track = 25
    num_per_test = 2

    # Run tests
    if step_size > 1:
        print "\nNOTE: Evaluating every %d-th EPD\n" % step_size

    for test in mode:
        print "Running %s STS test." % test
        # load STS epds
        epds = get_epds_by_mode(test)[::step_size]

        test_best = []
        test_worst = []

        # Test epds
        score = 0
        max_score = 0
        length = len(epds)
        print "STS: Scoring %s EPDS. Progress: " % length,
        print_perc = 5  # percent to print at
        for i, epd in enumerate(epds):
            # Print info
            count += 1
            if (i % (length / (100.0 / print_perc)) - 100.0 / length) < 0:
                print "%d%% " % (i / (length / 100.0)),
                sys.stdout.flush()

            board, move_scores = parse_epd(epd)

            # Get move
            move = player.get_move(board)

            # score
            max_score += 10

            try:
                new_score = move_scores[move]
            except KeyError:
                new_score = 0

            score += new_score

            test_best.append((move, new_score, epd))
            test_worst.append((move, new_score, epd))

            test_best.sort(reverse=True, key=lambda x: x[1])
            test_worst.sort(key=lambda x: x[1])

            test_best = test_best[:num_per_test]
            test_worst = test_worst[:num_per_test]

        best += test_best
        worst += test_worst

        best.sort(reverse=True, key=lambda x: x[1])
        worst.sort(key=lambda x: x[1])

        best = best[:num_track]
        worst = worst[:num_track]

        print ""

        # append
        scores.append(score)
        max_scores.append(max_score)

    # Highest scoring epds
    print "Evaluated on %d positions" % (count)
    if verbose:
        print "\n\n"
        wave = "¸,ø¤º°`°º¤ø,¸¸,ø¤º°¤ø,¸,ø¤º°`°º¤ø,¸¸,ø¤º°"
        reverse_wave = "°º¤ø,¸¸,ø¤º°`°º¤ø,¸,ø¤°º¤ø,¸¸,ø¤º°`°º¤ø,¸"
        print "\n%s  BEST SCORES  %s\n" % (wave, reverse_wave)
        title = ['Move', 'Score', 'EPD']
        print("\t\t{: <15} {: <15} {: <100}".format(*title))
        for result in best:
            print("\t\t{: <15} {: <15} {: <100}".format(*result))

        print "\n%s  WORST SCORES %s\n" % (wave, reverse_wave)
        print("\t\t{: <15} {: <15} {: <100}".format(*title))
        for result in worst:
            print("\t\t{: <15} {: <15} {: <100}".format(*result))

        print "\nSCORES BY TEST SUITE\n"
        print("\t\t{: <30} {: >30}".format(*["TEST SUITE NAME", "SCORE"]))
        for test, score, max_score in zip(mode, scores, max_scores):
            print("\t\t{: <30} {: >30}".format(*[test, '%d/%d' % (score, max_score)]))
        print ''

    return scores, max_scores


def parse_epd(epd):
    """
    Parses an EPD for use in STS. Returns the board and a move score dictionary.
    Input:
        epd [String]
            EPD describing the chess position.
    Output:
        fen, move_scores (Tuple)
            board [chess.Board]: Board as described by the EPD.
            move_scores [Dictionary]: Move score dictionary. Keys are Chess.Move objects.
    """

    board = chess.Board()

    # Set epd
    ops = board.set_epd(epd)

    # Parse move scores
    move_scores = dict()
    # print ops
    if 'c0' in ops:
        for m, s in [x.rstrip(',').split('=') for x in ops['c0'].split(' ')]:
            try:
                move_scores[board.parse_san(m)] = int(s)
            except ValueError:
                move_scores[board.parse_uci(m)] = int(s)
    else:
        move_scores[ops['bm'][0]] = 10

    return board, move_scores


def get_epds_by_mode(mode):
    """
    Returns the epds given an STS mode.
    Input:
        mode [List of Strings] or [String]
            Selects the test mode(s), see Teacher.eval_sts for options.
    Output:
        epds [List of Strings]
            List of epds.
    """

    if mode == 'strategy':
        epds = get_epds([resource_filename('guerilla', 'data/STS/' + f + '.epd')
                                 for f in sts_strat_files])
    elif mode == 'pieces':
        epds = get_epds([resource_filename('guerilla', 'data/STS/' + f + '.epd')
                                 for f in sts_piece_files])
    else:
        # Specific file
        try:
            epds = get_epds(resource_filename('guerilla', 'data/STS/' + mode + '.epd'))
        except IOError:
            raise ValueError("Error %s is an invalid test mode." % mode)

    return epds


def get_epds(filenames):
    """
    Returns a list of epds from the given file or list of files.
    Input:
        filename [String or List of Strings]
            Filename(s) of EPD file to open. Must include absolute path.
    Output:
        epds [List of Strings]
            List of epds.
    """
    if type(filenames) is not list:
        filenames = [filenames]

    epds = []
    for filename in filenames:
        with open(filename, 'r') as f:
            epds += [line.rstrip() for line in f]

    return epds


def sparse_test():
    with Guerilla('Harambe', load_file='4790.p',
                  search_type='minimax',
                  search_params={'max_depth': 1}) as g:
        # g.search.order_moves = True
        # g.search.order_function = material_balance
        # print eval_sts(g, mode=sts_piece_files)
        start = time.time()
        print eval_sts(g, mode=sts_strat_files, step_size=5)
        print time.time() - start


def main():
    with Guerilla('Harambe', search_type='minimax', search_params={'max_depth': 2}, load_file='6811.p') as g:
        print eval_sts(g, mode=sts_strat_files)

if __name__ == '__main__':
    # sparse_test()
    main()
