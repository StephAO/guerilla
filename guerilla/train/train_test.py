# Train Unit tests
import pickle
import random as rnd
import sys
import traceback

import chess
import numpy as np
import tensorflow as tf
from guppy import hpy
from pkg_resources import resource_filename

import guerilla.data_handler as dh
import guerilla.play.neural_net as nn
import guerilla.train.stockfish_eval as sf
import guerilla.train.sts as sts
import guerilla.train.chess_game_parser as cgp
from guerilla.players import Guerilla
from guerilla.train.teacher import Teacher

###############################################################################
# STOCKFISH TESTS
###############################################################################

def stockfish_test():
    """
    Tests stockfish scoring script and score mapping.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    seconds = 2
    max_attempts = 3

    # Fens in INCREASING score value
    fens = [None] * 8
    fens[0] = dh.flip_board('3qr1Qk/pbpp2pp/1p5N/6b1/2P1P3/P7/1PP2PPP/R4RK1 b - - 0 1')  # White loses in 2 moves
    fens[1] = dh.flip_board('3qr1Qk/pbpp2pp/1p5N/6b1/2P1P3/P7/1PP2PPP/R4RK1 b - - 0 1')  # White loses in 1 move
    fens[2] = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1'  # Starting position
    fens[3] = 'r1bqkbnr/ppp2ppp/2p5/2Q1p3/4P3/3P1N2/PPP2PPP/RNB1K2R w - - 0 5'  # Midgame where white is winning
    fens[4] = '2qrr1n1/3b1kp1/2pBpn1p/1p2PP2/p2P4/1BP5/P3Q1PP/4RRK1 w - - 0 1'  # mate in 10 (white wins)
    fens[5] = '4Rnk1/pr3ppp/1p3q2/5NQ1/2p5/8/P4PPP/6K1 w - - 0 1'  # mate in 3 (white wins)
    fens[6] = '3qr2k/pbpp2pp/1p5N/3Q2b1/2P1P3/P7/1PP2PPP/R4RK1 w - - 0 1'  # mate in 2 (white wins)
    fens[7] = '3q2rk/pbpp2pp/1p5N/6b1/2P1P3/P7/1PP2PPP/R4RK1 w - - 0 1'  # mate in 1 (white wins)

    # Test white play next
    prev_score = float('-inf')
    for i, fen in enumerate(fens):
        score = sf.get_stockfish_score(fen, seconds=seconds, num_attempt=max_attempts)
        if score < prev_score:
            print "Failure: Fen (%s) scored %f while fen (%s) scored %f. The former should have a lower score." \
                  % (fens[i - 1], prev_score, fens[i], score)
            return False
        if sf.sigmoid_array(score) < sf.sigmoid_array(prev_score):
            print "Failure: Fen (%s) scored %f while fen (%s) scored %f. The former should have a lower score." \
                  % (fens[i - 1], sf.sigmoid_array(prev_score), fens[i], sf.sigmoid_array(score))
            return False
        prev_score = score

    # Test black play next
    prev_score = float('-inf')
    for i, fen in enumerate(fens):
        score = sf.get_stockfish_score(fen, seconds=seconds, num_attempt=max_attempts)
        if score < prev_score:
            print "Failure: Fen (%s) scored %f while fen (%s) scored %f. The former should have a lower score." \
                  % (dh.flip_board(fens[i - 1]), prev_score, dh.flip_board(fens[i]), score)
            return False
        if sf.sigmoid_array(score) < sf.sigmoid_array(prev_score):
            print "Failure: Fen (%s) scored %f while fen (%s) scored %f. The former should have a lower score." \
                  % (dh.flip_board(fens[i - 1]), sf.sigmoid_array(prev_score), dh.flip_board(fens[i]),
                     sf.sigmoid_array(score))
            return False
        prev_score = score

    return True

def nsv_test(num_check=40, max_step=10000, tolerance=2e-2, allow_err=0.3, score_repeat=3):
    """
    Tests that fens.nsv and sf_values.nsv file are properly aligned. Also checks that the FENS are "white plays next".
    NOTE: Need at least num_check*max_step stockfish and fens stored in the nsv's.
    Input:
        num_check [Int]
            Number of fens to check.
        max_step [Int]
            Maximum size of the random line jump within a file. i.e. at most, the next line checked will be max_step
            lines away from the current line.
        tolerance [Float]
            How far away the expected stockfish score can be from the actual stockfish score. 0 < tolerance < 1
        allow_err [Float]
            The percentage of mismatching stockfish scores allowed. 0 < allow_err < 1
        score_repeat [Int]
            Each stockfish scoring is repeated score_repeat times and the median is taken. Allows for variations in
            available memory.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    # Number of seconds spent on each stockfish score
    seconds = 1
    wrong = []
    max_wrong = num_check * allow_err

    with open(resource_filename('guerilla', 'data/extracted_data/fens.nsv'), 'r') as fens_file, \
            open(resource_filename('guerilla', 'data/extracted_data/sf_values.nsv'), 'r') as sf_file:
        fens_count = 0
        line_count = 0
        while fens_count < num_check and len(wrong) <= max_wrong:
            for i in range(rnd.randint(0, max_step)):
                fens_file.readline()
                sf_file.readline()
                line_count += 1

            # Get median stockfish score
            fen = fens_file.readline().rstrip()
            median_arr = []
            for i in range(score_repeat):
                median_arr.append(sf.get_stockfish_score(fen, seconds=seconds))

            # Convert to probability of winning
            expected = sf.sigmoid_array(np.median(median_arr))
            actual = float(sf_file.readline().rstrip())

            if abs(expected - actual) > tolerance:
                wrong.append("For FEN '%s' calculated score of %f, got file score of %f (line %d)." %
                             (fen, expected, actual, line_count))

            if dh.black_is_next(fen):
                print "White does not play next in this FEN: %s" % fen
                return False

            fens_count += 1
            line_count += 1

    if len(wrong) > max_wrong:
        for info in wrong:
            print info

        return False

    return True

###############################################################################
# TRAINING TESTS
###############################################################################

def training_test(nn_input_type, verbose=False):
    """
    Runs training in variety of fashions.
    Checks crashing, decrease in cost over epochs, consistent output, and memory usage.
    """
    success = True
    # Set hyper params for mini-test
    
    for t_m in nn.NeuralNet.training_modes:
        error_msg = ""
        try:
            with Guerilla('Harambe', 'w', verbose=verbose, NN_INPUT_TYPE=nn_input_type) as g:
                g.search.max_depth = 1

                t = Teacher(g, training_mode=t_m, test=True, verbose=verbose, 
                            hp_load_file='training_test.yaml')
                if t_m == 'adagrad':
                    t.set_hyper_params(LEARNING_RATE=0.00001)
                elif t_m == 'adadelta':
                    continue  # TODO remove when adadelta is fully implemented
                    t.set_hyper_params(LEARNING_RATE=0.00001)
                elif t_m == 'bootstrap':
                    t.set_hyper_params(LEARNING_RATE=0.00001)

                t.set_bootstrap_params(num_bootstrap=400)  # 488037
                t.set_td_params(num_end=3, num_full=3, randomize=False, end_length=3, full_length=3, batch_size=5)
                t.set_sp_params(num_selfplay=1, max_length=3)
                t.sts_on = False
                t.sts_interval = 100

                pre_heap_size = hpy().heap().size
                t.run(['train_bootstrap', 'train_td_end', 'train_td_full', 'train_selfplay'], training_time=60)
                post_heap_size = hpy().heap().size

                loss = pickle.load(open(resource_filename('guerilla', 'data/loss/loss_test.p'), 'rb'))
                # Wrong number of losses
                if len(loss['train_loss']) != t.hp['NUM_EPOCHS'] + 1 or len(loss['loss']) != t.hp['NUM_EPOCHS'] + 1:
                    error_msg += "Some bootstrap epochs are missing training or validation losses.\n" \
                                 "Number of epochs: %d,  Number of training losses: %d, Number of validation losses: %d\n" % \
                                 (t.hp['NUM_EPOCHS'], len(loss['train_loss']), len(loss['loss']))
                    success = False
                # Training loss went up
                if loss['train_loss'][0] <= loss['train_loss'][-1]:
                    error_msg += "Bootstrap training loss went up. Losses:\n%s\n" % (loss['train_loss'])
                    success = False
                # Validation loss went up
                if loss['loss'][0] <= loss['loss'][-1]:
                    error_msg += "Bootstrap validation loss went up. Losses:\n%s\n" % (loss['loss'])
                    success = False
                # Memory usage increased significantly
                if float(abs(post_heap_size - pre_heap_size)) / float(pre_heap_size) > 0.01:
                    success = False
                    error_msg += "Memory increasing significantly when running training.\n" \
                                 "Starting heap size: %d bytes, Ending heap size: %d bytes. Increase of %f %%\n" \
                                 % (pre_heap_size, post_heap_size,
                                    100. * float(abs(post_heap_size - pre_heap_size)) / float(pre_heap_size))
        # Training failed
        except Exception:
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_msg += "The following error occured during the training:" \
                         "\n  type:\n    %s\n\n  Error msg:\n    %s\n\n  traceback:\n    %s\n" % \
                         (str(exc_type).split('.')[1][:-2], exc_value, \
                          '\n    '.join(''.join(traceback.format_tb(exc_traceback)).split('\n')))
            success = False

        if not success:
            print "Training with type %s fails:\n%s" % (t_m, error_msg)

    return success

def learn_sts_test(nn_input_type, mode='strategy', thresh=0.9):
    """
    NOTE: DEPRECATED. The test does not converge quickly (if at all). Replaced with learn_moves_test.

    Tests that Guerilla can learn the best moves in the Strategic Test Suite (STS).
    Fetches all the STS epds. Takes the top moves and gives them high probability of winning.
    Then randomly generates the same number of bad moves and gives them a low probability of winning.
    Trains the guerilla on this set of data. Runs STS on the guerilla trained on this set of data.
    Should yield a high STS score, the test is succesful if the STS score is sufficiently high.
    Note:
        Conversion to probability is winning done by the function P(x) = 0.6 + x^2*(0.003) where x in [0, 10].
    Input:
        mode [String]
            STS mode(s) to train and test on. See Guerilla.Teacher.eval_sts()  documentation for options.
        thresh [Float]
            Minimum STS score (as a percent of the maximum score) necessary for the test to be considered a success.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    # Get EPDS and Scores
    if type(mode) is not list:
        mode = [mode]

    # vars
    board = chess.Board()

    # Run tests
    epds = []
    for test in mode:
        epds += sts.get_epds_by_mode(test)

    # Convert scores to probability of winning and build data
    fens = []
    values = []
    for i, epd in enumerate(epds):

        board, move_scores = sts.parse_epd(epd)

        for move, score in move_scores.iteritems():

            # Add good moves
            board.push(move) # apply move
            fen = board.fen()
            fens.append(fen if dh.white_is_next(fen) else dh.flip_board(fen))
            values.append(1 if score == 10 else 0)
            board.pop() # undo move

        # # Add random bad move
        # move = rnd.choice(list(set(board.legal_moves) - set(move_scores.iterkeys())))
        # board.push(move)  # apply move
        # fen = board.fen()
        # fens.append(fen if dh.white_is_next(fen) else dh.flip_board(fen))
        # values.append(0.25)
        # board.pop() # undo move

    print len(fens)

    # Set hyper parameters
    hp = {}
    hp['NUM_EPOCHS'] = 50
    hp['BATCH_SIZE'] = 10
    hp['VALIDATION_SIZE'] = 50
    hp['TRAIN_CHECK_SIZE'] = 10
    hp['LEARNING_RATE'] = 0.0001
    hp['LOSS_THRESHOLD'] = -100 # Make it so it never stops by convergence since VALIDATION_SIZE = 0

    # Add extra evaluation boards
    fens += fens[-hp['VALIDATION_SIZE']:]
    values += values[-hp['VALIDATION_SIZE']:]

    # set to multiple of batch size
    if (len(fens) % hp['BATCH_SIZE']) != 0:
        fens = fens[:(-1) * (len(fens) % hp['BATCH_SIZE'])]
    values = values[:len(fens)]

    # Train and Test Guerilla
    with Guerilla('Harambe', 'w', NN_INPUT_TYPE=nn_input_type) as g:
        g.search.max_depth = 1
        # Train
        t = Teacher(g, training_mode='adagrad')
        t.set_hyper_params(**hp)
        t.train_bootstrap(fens, values)

        # Run STS Test
        result = sts.eval_sts(g, mode=mode)

    if float(result[0][0])/result[1][0] <= thresh:
        print "STS Scores was too low, got a score of %d/%d" % (result[0][0], result[1][0])
        return False

    return True


def learn_moves_test(nn_input_type, num_test=3, num_attempt=3, verbose=False):
    """
    Tests that Guerilla can learn the best moves of a few boards, thus demonstrating that the input Guerilla converge
    to learning chess moves.
    Details: For each board scores a given move highly (called the goal move) and the others poorly.
    Trains the guerilla on this set of data. Then sees if Guerilla plays that move when given the board as an input.
    Input:
        mode [String] (Optiona)
            Number of boards to learn moves on and try to play correctly. Higher makes it harder for the test to pass.
        num_attempt [Int] (Optional)
            The number of attempts to make in converging to the moves.
            Sometimes the random weight initialization is unlucky.
        verbose [Boolean]
            Turn Verbose Mode on and off.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    # Set hyper parameters
    hp = {}
    hp['NUM_EPOCHS'] = 30
    hp['BATCH_SIZE'] = 10
    hp['VALIDATION_SIZE'] = 30
    hp['TRAIN_CHECK_SIZE'] = 10
    hp['LEARNING_RATE'] = 0.00005
    hp['LOSS_THRESHOLD'] = 0.001

    # Probability value Constants (0 <= x <= 1)
    high_value = 0.9
    low_value = 0.1

    # Load fens
    fen_multiplier = 20
    spacing = 100
    base_fens = cgp.load_fens(num_values=num_test*spacing)[::spacing] # So not all within the same game

    # For each fen get all moves, score one move's board highly (goal move), the others poorly
    goal_moves = [] # List of tuples
    fens = []
    values = []
    val_fens = []
    val_values = []
    for fen in base_fens:
        board = chess.Board(fen)
        moves = board.legal_moves

        # Store goal_move info for scoring
        goal_move = np.random.choice(list(moves))
        goal_moves.append((fen, goal_move))

        # Score goal move highly
        board.push(goal_move)
        fens += [dh.flip_board(board.fen())]*fen_multiplier # Flip board and give low value since NN input must be white next
        values += [1 - high_value]*fen_multiplier
        board.pop()

        # Build validation set
        val_fens += [fens[-1]] * (hp['VALIDATION_SIZE']/num_test)
        val_values += [1.0 - high_value] * (hp['VALIDATION_SIZE']/num_test)  # values[-hp['VALIDATION_SIZE']:]

        # score other moves poorly
        for move in (set(board.legal_moves) - {goal_move}):
            board.push(move)
            fens += [dh.flip_board(board.fen())] # Flip board and give high value since NN input must be white next
            values += [1 - low_value]
            board.pop()

    # set to multiple of batch size
    if (len(fens) % hp['BATCH_SIZE']) != 0:
        fens = fens[:(-1) * (len(fens) % hp['BATCH_SIZE'])]
    values = values[:len(fens)]

    # Combine validation set
    fens += val_fens
    values += val_values

    # Train and Test Guerilla
    err_msg = ''
    for i in range(num_attempt):
        score = 0
        with Guerilla('Harambe', 'w', verbose=verbose, NN_INPUT_TYPE=nn_input_type) as g:
            g.search.max_depth = 1
            # Train
            t = Teacher(g, training_mode='gradient_descent', verbose=verbose)
            t.set_hyper_params(**hp)
            t.train_bootstrap(fens, values)

            # Evaluate
            for fen, goal_move in goal_moves:
                board = chess.Board(fen)
                result_move = g.get_move(board)

                if result_move == goal_move:
                    score += 1
                else:
                    board.push(goal_move)
                    goal_score = 1 - g.nn.evaluate(dh.flip_board(board.fen()))
                    board.pop()
                    board.push(result_move)
                    result_score = 1 - g.nn.evaluate(dh.flip_board(board.fen()))
                    err_msg += ('FAILURE: Learn Move Mismatch: Expected %s got %s \n Neural Net Scores: %s - > %f, %s -> %f\n' %
                                    (goal_move, result_move, goal_move, goal_score, result_move, result_score))

        if score == num_test:
            return True

        err_msg += "Failed attempt #%s...\n" % i

    print err_msg
    return False

def load_and_resume_test(nn_input_type, verbose=False):
    """
    Tests the load_and_resume functionality of teacher.
    Things it checks for:
        (1) Doesn't crash.
        (2) Weights are properly loaded.
        (3) Graph training variables are properly loaded.
        (4) Correct action is loaded.
        (5) Correct set of actions is loaded.
        (6) Correct number of epochs.
    Does not check (among other things):
        (-) That all the necessary components of the training state are stored.
        (-) That the correct sequence of training actions is taken.
        (-) That training reduces the loss.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """

    # Modify hyperparameters for a small training example.
    success = True
    hp = {}
    hp['NUM_EPOCHS'] = 5
    hp['BATCH_SIZE'] = 5
    hp['VALIDATION_SIZE'] = 5
    hp['TRAIN_CHECK_SIZE'] = 5
    hp['TD_LRN_RATE'] = 0.00001  # Learning rate
    hp['TD_DISCOUNT'] = 0.7  # Discount rate
    hp['LEARNING_RATE'] = 0.00001

    # Pickle path
    loss_path = resource_filename('guerilla', 'data/loss/')

    # Test for each training type & all training types together
    train_actions = Teacher.actions[:-1]
    train_actions.append(Teacher.actions[:-1])
    for action in train_actions:
        set_of_actions = action if isinstance(action, list) else [action]

        # Error message:
        error_msg = ''

        # Reset graph
        tf.reset_default_graph()

        # Run action
        with Guerilla('Harambe', 'w', verbose=verbose, NN_INPUT_TYPE=nn_input_type) as g:
            g.search.max_depth = 1
            t = Teacher(g, test=True, verbose=verbose)
            t.set_hyper_params(**hp)
            t.set_bootstrap_params(num_bootstrap=50)  # 488037
            t.set_td_params(num_end=3, num_full=3, randomize=False, end_length=2, full_length=2)
            t.set_sp_params(num_selfplay=3, max_length=5)

            # Run
            t.run(set_of_actions, training_time= (0.5 if not isinstance(action, list) else 4))

            # Save current action
            pause_action = t.actions[t.curr_action_idx]

            # Save Weights
            weights = g.nn.get_weight_values()

            # Save graph training variables
            train_vars = g.nn.sess.run(g.nn.get_training_vars())

        # Reset graph
        tf.reset_default_graph()

        # Run resume
        with Guerilla('Harambe', 'w', verbose=verbose, NN_INPUT_TYPE=nn_input_type) as g:
            g.search.max_depth = 1
            t = Teacher(g, test=True, verbose=verbose)
            t.set_hyper_params(**hp)
            t.set_bootstrap_params(num_bootstrap=50)  # 488037

            # Run
            t.run(['load_and_resume'])

            # Save loaded current action
            state = t.load_state() # resets weights and training vars to start of resume values
            new_actions = state['actions']
            new_action = new_actions[state['curr_action_idx']]

            # Get new weights
            new_weights = g.nn.get_weight_values()

            # Save new training variables
            new_train_vars = g.nn.sess.run(g.nn.get_training_vars())


        # Compare weight values
        result_msg = dh.diff_dict_helper(weights, new_weights)
        if result_msg:
            error_msg += "Weight did not match.\n"
            error_msg += result_msg
            success = False

        # Compare graph training variable values
        result_msg = dh.diff_dict_helper(train_vars, new_train_vars)
        if result_msg:
            error_msg += "Training variables did not match.\n"
            error_msg += result_msg
            success = False

        # Compare the action
        if pause_action != new_action:
            error_msg += "Current action was not saved and loaded properly. \n"
            error_msg += "Saved:\n %s \n Loaded:\n %s\n" % (pause_action, new_action)
            success = False

        # Compare the set of actions
        if set_of_actions != new_actions:
            error_msg += "Set of actions was not saved and loaded properly. \n"
            error_msg += "Saved:\n %s \n Loaded:\n %s\n" % (str(set_of_actions), str(new_actions))
            success = False

        # Check that correct number of epochs is run
        with open(loss_path + 'loss_test.p', 'r') as f:
            loss = pickle.load(f)
            if hp['NUM_EPOCHS'] != (len(loss['loss']) - 1):
                error_msg += "On action %s there was the wrong number of epochs. " % action
                error_msg += "Expected %d epochs, but got %d epochs." % (hp['NUM_EPOCHS'], len(loss['loss']) - 1)
                success = False

        if not success:
            print "Load and resume with action %s fails:\n%s" % (str(action), error_msg)

    return success

def run_train_tests():
    all_tests = {}
    all_tests["Stockfish Tests"] = {
        #'Stockfish Handling': stockfish_test,
        #'NSV Alignment': nsv_test
                                    }

    all_tests["Training Tests"] = {
        'Training': training_test,
        'Load and Resume': load_and_resume_test,
        'Learn Moves': learn_moves_test
    }

    

    success = True
    input_types = ['movemap', 'bitmap', 'giraffe',]
    print "\nRunning Train Tests...\n"

    print "--- Stockfish tests ---"
    for name, test in all_tests["Stockfish Tests"].iteritems():
        print "Testing " + name + "..."
        if not test():
            print "%s test failed" % name.capitalize()
            success = False

    print "--- Training Tests ---"
    for it in input_types:
        print "Testing using input type", it.upper()
        for name, test in all_tests["Training Tests"].iteritems():
            print "Testing " + name + "..."
            if not test(it):
                print "%s test failed" % name.capitalize()
                success = False

    return success

def main():
    if run_train_tests():
        print "All tests passed"
    else:
        print "You broke something - go fix it"

if __name__ == '__main__':
    main()