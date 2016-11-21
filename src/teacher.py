import os
import sys
import random
import numpy as np
import tensorflow as tf
import math
import chess.pgn
import chess
import time
import matplotlib.pyplot as plt
import pickle
import copy
from players import Player, Guerilla
from operator import add

import data_handler as dh
import stockfish_eval as sf
import chess_game_parser as cgp
from hyper_parameters import *


class Teacher:
    actions_dict = [
        'train_bootstrap',
        'train_td_endgames',
        'train_td_full',
        'train_selfplay',
        'load_and_resume'
    ]

    def __init__(self, _guerilla, test=False, verbose=True):

        # dictionary of different training/evaluation methods

        self.guerilla = _guerilla
        self.nn = _guerilla.nn
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.start_time = None
        self.training_time = None
        self.actions = None
        self.curr_action_idx = None
        self.saved = None

        self.test = test
        self.verbose = verbose

        # Bootstrap parameters
        self.num_bootstrap = -1

        # TD-Leaf parameters
        self.td_pgn_folder = self.dir_path + '/../helpers/pgn_files/single_game_pgns'
        self.td_rand_file = False  # If true then TD-Leaf randomizes across the files in the folder.
        self.td_num_endgame = -1  # The number of endgames to train on using TD-Leaf (-1 = All)
        self.td_num_full = -1  # The number of full games to train on using TD-Leaf
        self.td_end_length = 12  # How many moves are included in endgame training
        self.td_full_length = -1  # Maximum number of moves for full game training (-1 = All)
        self.td_w_update = None 
        self.td_fen_index = 0
        self.td_batch_size = 50

        if self.nn.training_mode == 'adagrad':
            self.td_adagrad_acc = None
            self.weight_shapes = []
            for weight in self.nn.all_weights:
                self.weight_shapes.append(self.nn.sess.run(tf.shape(weight)))

        # Self-play parameters
        self.sp_num = 1  # The number of games to play against itself
        self.sp_length = 12  # How many moves are included in game playing

        # STS Evaluation Parameters
        self.sts_on = False  # Whether STS evaluation occurs during training
        self.sts_interval = 50  # Interval at which STS evaluation occurs, unit is number of games
        self.sts_mode = "strategy"  # STS mode for evaluation
        self.sts_depth = self.guerilla.search.max_depth  # Depth used for STS evaluation (can override default)

    # ---------- RUNNING AND RESUMING METHODS

    def run(self, actions, training_time=None):
        """ 
            1. load data from file
            2. configure data
            3. run actions
        """
        self.start_time = time.time()
        self.training_time = training_time
        self.actions = actions
        self.curr_action_idx = 0  # This value gets modified if resuming
        self.saved = False

        if self.actions[0] == 'load_and_resume':
            self.resume(training_time)

        # Note: This cannot be a for loop as self.curr_action_idx gets set to non-zero when resuming.
        while True:
            # Save new weight values if necessary
            if not self.saved and self.curr_action_idx > 0 and not self.test:
                weight_file = "weights_" + self.actions[self.curr_action_idx - 1] \
                              + "_" + time.strftime("%Y%m%d-%H%M%S") + ".p"
                self.nn.save_weight_values(_filename=weight_file)

            # Check if done
            if self.curr_action_idx >= len(self.actions):
                break
            elif self.out_of_time():
                # Check if run out of time between actions, save if necessary
                if not self.saved:
                    self.save_state(state={})
                break

            # Else
            action = self.actions[self.curr_action_idx]
            if action == 'train_bootstrap':
                if self.verbose:
                    print "Performing Bootstrap training!"
                    print "Fetching stockfish values..."

                fens = cgp.load_fens(num_values=self.num_bootstrap)
                if (len(fens) % hp['BATCH_SIZE']) != 0:
                    fens = fens[:(-1) * (len(fens) % hp['BATCH_SIZE'])]
                true_values = sf.load_stockfish_values(num_values=len(fens))

                self.train_bootstrap(fens, true_values)
            elif action == 'train_td_endgames':
                if self.verbose:
                    print "Performing endgame TD-Leaf training!"
                self.train_td(True)
            elif action == 'train_td_full':
                if self.verbose:
                    print "Performing full-game TD-Leaf training!"
                self.train_td(False)
            elif action == 'train_selfplay':
                if self.verbose:
                    print "Performing self-play training!"
                self.train_selfplay()
            elif action == 'load_and_resume':
                raise ValueError("Error: Resuming must be the first action in an action set.")
            else:
                raise NotImplementedError("Error: %s is not a valid action." % action)

            self.curr_action_idx += 1

    def save_state(self, state, filename="state.p"):
        """
            Save current state so that training can resume at another time
            Note: Can only save state at whole batch intervals (but mid-epoch is fine)
            Inputs:
                state[dict]:
                    For bootstrap:
                        game_indices[list of ints]
                        (if action == 'train_bootstrap')
                            epoch_num[int]
                            loss[list of floats]
                        (if action == 'train_td_...')
                            start_idx [int]
                filename[string]:
                    filename to save pickle to
        """
        state['curr_action_idx'] = self.curr_action_idx
        state['actions'] = self.actions

        # Save training parameters
        state['td_leaf_param'] = {'randomize': self.td_rand_file,
                                  'num_end': self.td_num_endgame,
                                  'num_full': self.td_num_full,
                                  'end_length': self.td_end_length,
                                  'full_length': self.td_full_length,
                                  'batch_size': self.td_batch_size,
                                  }
        if self.nn.training_mode == 'adagrad':
            state['adagrad'] = {'w_update': self.td_w_update,
                                'fen_index': self.td_fen_index,
                                'adagrad_acc': self.td_adagrad_acc}

        state['sp_param'] = {'num_selfplay': self.sp_num, 'max_length': self.sp_length}

        # Save STS evaluation parameters
        state['sts_on'] = self.sts_on
        state['sts_interval'] = self.sts_interval
        state['sts_mode'] = self.sts_mode
        state['sts_depth'] = self.sts_depth

        # Save training variables
        train_var_path = self.dir_path + '/../pickles/train_vars/in_training_vars.vars'
        train_var_file = self.nn.save_training_vars(train_var_path)
        if train_var_file:
            state['train_var_file'] = train_var_file

        pickle_path = self.dir_path + '/../pickles/' + filename
        self.nn.save_weight_values(_filename='in_training_weight_values.p')
        pickle.dump(state, open(pickle_path, 'wb'))
        self.saved = True
        if self.verbose:
            print "State saved."

    def load_state(self, filename='state.p'):
        """
            Load state from pickle. Returns information necessary to resume training
            Inputs:
                filename[string]:
                    filename to load pickle from
            Outputs:
                state[dict]:
                    action[string]
                    epoch_num[int]
                    remaining_boards[list of ints]
                    loss[list of floats]
        """
        pickle_path = self.dir_path + '/../pickles/' + filename
        state = pickle.load(open(pickle_path, 'rb'))

        # Load training parameters
        self.set_td_params(**state.pop('td_leaf_param'))
        self.set_sp_params(**state.pop('sp_param'))

        # Load adagrad params
        if self.nn.training_mode == 'adagrad':
            self.td_w_update = state['adagrad']['w_update']
            self.td_fen_index = state['adagrad']['fen_index']
            self.td_adagrad_acc = state['adagrad']['adagrad_acc']

        # Load STS evaluation parameters
        self.sts_on = state['sts_on']
        self.sts_interval = state['sts_interval']
        self.sts_mode = state['sts_mode']
        self.sts_depth = state['sts_depth']

        # Load training variables
        if 'train_var_file' in state:
            self.nn.load_training_vars(state['train_var_file'])

        self.curr_action_idx = state['curr_action_idx']
        self.actions = state['actions'] + self.actions[1:]
        self.nn.load_weight_values(_filename='in_training_weight_values.p')

        return state

    def resume(self, training_time=None):
        """
            Resumes training from a previously paused training session
        """
        if self.verbose:
            print "Resuming training"
        state = self.load_state()

        self.start_time = time.time()
        self.training_time = training_time

        if 'game_indices' not in state:
            # Stopped between actions.
            return

        action = self.actions[self.curr_action_idx]

        if action == 'train_bootstrap':
            if self.verbose:
                print "Resuming Bootstrap training..."

            fens = cgp.load_fens(num_values=self.num_bootstrap)
            if (len(fens) % hp['BATCH_SIZE']) != 0:
                fens = fens[:(-1) * (len(fens) % hp['BATCH_SIZE'])]

            true_values = sf.load_stockfish_values(num_values=len(fens))

            # finish epoch
            train_check_spacing = (len(fens) - hp['VALIDATION_SIZE']) / hp['TRAIN_CHECK_SIZE']

            train_fens = fens[:(-1) * hp['VALIDATION_SIZE']]  # fens to train on
            valid_fens = fens[(-1) * hp['VALIDATION_SIZE']:]  # fens to check convergence on
            train_check_fens = train_fens[::train_check_spacing]  # fens to evaluate training error on

            train_values = true_values[:(-1) * hp['VALIDATION_SIZE']]
            valid_values = true_values[(-1) * hp['VALIDATION_SIZE']:]
            train_check_values = train_values[::train_check_spacing]

            self.weight_update_bootstrap(train_fens, train_values, state['game_indices'], self.nn.train_step)

            # evaluate nn for convergence # TODO: Fix duplicate loss being stored
            state['loss'].append(self.evaluate_bootstrap(valid_fens, valid_values))
            state['train_loss'].append(self.evaluate_bootstrap(train_check_fens, train_check_values))
            curr_loss = state['loss'][-2] - state['loss'][-1]
            base_loss = state['loss'][0] - state['loss'][1]
            if False: # TODO pick better convergence threshold
                self.nn.save_weight_values()
                plt.plot(range(state['epoch_num']), state['loss'])
                plt.show()

            # continue with rests of epochs
            self.train_bootstrap(fens, true_values, start_epoch=state['epoch_num'],
                                 loss=state['loss'], train_loss=state['train_loss'])
        elif action == 'train_td_endgames':
            if self.verbose:
                print "Resuming endgame TD-Leaf training..."
            self.train_td(True, game_indices=state['game_indices'], start_idx=state['start_idx'],
                          sts_scores=state['sts_scores'])
        elif action == 'train_td_full':
            if self.verbose:
                print "Resuming full-game TD-Leaf training..."
            self.train_td(False, game_indices=state['game_indices'], start_idx=state['start_idx'],
                          sts_scores=state['sts_scores'])
        elif action == 'train_selfplay':
            if self.verbose:
                print "Resuming self-play training..."
            self.train_selfplay(game_indices=state['game_indices'], start_idx=state['start_idx'],
                                sts_scores=state['sts_scores'])
        elif action == 'load_and_resume':
            raise ValueError("Error: It's trying to resume on a resume call - This shouldn't happen.")
        else:
            raise NotImplementedError("Error: %s is not a valid action." % action)

        self.curr_action_idx += 1
        return

    def out_of_time(self):
        """
        Returns True if training has run out of time. False otherwise
            Output:
                [Boolean]
        """
        return self.training_time is not None and time.time() - self.start_time >= self.training_time

    # ---------- BOOTSTRAP TRAINING METHODS

    def set_bootstrap_params(self, num_bootstrap=None):
        self.num_bootstrap = num_bootstrap

    def train_bootstrap(self, fens, true_values, start_epoch=0, loss=None, train_loss=None):
        """
            Train neural net

            Inputs:
                fens[list of strings]:
                    The fens representing the board states
                true_values[ndarray]:
                    Expected output for each chess board state (between 0 and 1)
        """

        train_check_spacing = (len(fens) - hp['VALIDATION_SIZE']) / hp['TRAIN_CHECK_SIZE']

        train_fens = fens[:(-1) * hp['VALIDATION_SIZE']]  # fens to train on
        valid_fens = fens[(-1) * hp['VALIDATION_SIZE']:]  # fens to check convergence on
        train_check_fens = train_fens[::train_check_spacing]  # fens to evaluate training error on

        train_values = true_values[:(-1) * hp['VALIDATION_SIZE']]
        valid_values = true_values[(-1) * hp['VALIDATION_SIZE']:]
        train_check_values = train_values[::train_check_spacing]

        num_boards = len(train_fens)

        if not loss:
            loss = []

        if not train_loss:
            train_loss = []

        # usr_in = raw_input("This will overwrite your old weights\' pickle, do you still want to proceed (y/n)?: ")
        # if usr_in.lower() != 'y':
        #    return
        if self.verbose:
            print "Training data on %d positions. Will save weights to pickle" % num_boards
            print "%16s %16s %16s " % ('Epoch', 'Validation Loss', 'Training Loss')
        loss.append(self.evaluate_bootstrap(valid_fens, valid_values))
        train_loss.append(self.evaluate_bootstrap(train_check_fens, train_check_values))
        if self.verbose:
            print "%16d %16.2f %16.2f" % (start_epoch, loss[-1], train_loss[-1])
        for epoch in xrange(start_epoch, hp['NUM_EPOCHS']):
            # Configure data (shuffle fens -> fens to channel -> group batches)
            game_indices = range(num_boards)
            random.shuffle(game_indices)

            # update weights
            save = self.weight_update_bootstrap(train_fens, train_values, game_indices, self.nn.train_step)

            # save state if timeout
            if save[0]:
                save[1]['epoch_num'] = epoch + 1
                save[1]['loss'] = loss
                save[1]['train_loss'] = train_loss
                self.save_state(save[1])
                return

            # evaluate nn for convergence
            loss.append(self.evaluate_bootstrap(valid_fens, valid_values))
            train_loss.append(self.evaluate_bootstrap(train_check_fens, train_check_values))
            if self.verbose:
                print "%16d %16.2f %16.2f" % (epoch + 1, loss[-1], train_loss[-1])

            if len(loss) > 2:
                base_loss = loss[0] - loss[1]
                curr_loss = loss[-2] - loss[-1]
                if False: # TODO pick better convergence threshold
                    if self.verbose:
                        print "Training complete: Reached convergence threshold"
                    break
        else:
            if self.verbose:
                print "Training complete: Reached max epoch, no convergence yet"

        if self.test:
            filename = self.dir_path + '/../pickles/loss_test.p'
        else:
            filename = self.dir_path + '/../pickles/loss_' + time.strftime("%Y%m%d-%H%M%S") + ".p"
            # save loss
        pickle.dump({"loss": loss, "train_loss": train_loss},
                    open(filename, 'wb'))
        # plt.plot(range(epoch + 1), error)
        # plt.show()

        return

    def weight_update_bootstrap(self, fens, true_values_, game_indices, train_step):
        """ Weight update for multiple batches"""

        if len(game_indices) % hp['BATCH_SIZE'] != 0:
            raise Exception("Error: number of fens provided (%d) is not a multiple of batch_size (%d)" %
                            (len(game_indices), hp['BATCH_SIZE']))

        num_batches = int(len(game_indices) / hp['BATCH_SIZE'])

        board_num = 0
        boards = np.zeros((hp['BATCH_SIZE'], 8, 8, hp['NUM_CHANNELS']))
        diagonals = np.zeros((hp['BATCH_SIZE'], 10, 8, hp['NUM_CHANNELS']))
        true_values = np.zeros(hp['BATCH_SIZE'])

        for i in xrange(num_batches):
            # if training time is up, save state
            if self.out_of_time():
                if self.verbose:
                    print "Bootstrap Timeout: Saving state and quitting"
                return True, {'game_indices': game_indices[(i * hp['BATCH_SIZE']):]}

            # set up batch
            for j in xrange(hp['BATCH_SIZE']):
                boards[j] = dh.fen_to_channels(fens[game_indices[board_num]])
                diagonals[j] = dh.get_diagonals(boards[j])
                true_values[j] = true_values_[game_indices[board_num]]
                board_num += 1

            # train batch
            self.nn.sess.run([train_step], feed_dict={self.nn.data: boards, self.nn.data_diags: diagonals,
                                                      self.nn.true_value: true_values})

        return False, {}

    def evaluate_bootstrap(self, fens, true_values):
        """
            Evaluate neural net

            Inputs:
                fens[list of strings]:
                    The fens representing the board states
                true_values[ndarray]:
                    Expected output for each chess board state (between 0 and 1)
        """

        # Configure data
        boards = np.zeros((hp['VALIDATION_SIZE'], 8, 8, hp['NUM_CHANNELS']))
        diagonals = np.zeros((hp['VALIDATION_SIZE'], 10, 8, hp['NUM_CHANNELS']))
        for i in xrange(hp['VALIDATION_SIZE']):
            boards[i] = dh.fen_to_channels(fens[i])
            diagonals[i] = dh.get_diagonals(boards[i])

        # Get loss
        error = self.nn.sess.run(self.nn.MAE, feed_dict={
            self.nn.data: boards,
            self.nn.data_diags: diagonals,
            self.nn.true_value: true_values
        })

        return error

    # ---------- TD-LEAF TRAINING METHODS

    # TODO: Handle complete fens format

    def set_td_params(self, num_end=None, num_full=None, randomize=None, pgn_folder=None,
                      end_length=None, full_length=None, batch_size=None):
        """
        Set the parameters for TD-Leaf.
            Inputs:
                num_end [Int]
                    Number of endgames to train on using TD-Leaf.
                num_full [Int]
                    Number of full games to train on using TD-Leaf.
                randomize [Boolean]
                    Whether or not to randomize across the pgn files.
                pgn_folder [String]
                    Folder containing chess games in PGN format.
                end_depth [Int]
                    Length of endgames.
                full_depth [Int]
                    Maximum length of full games. (-1 for no max)
        """
        if num_end:
            self.td_num_endgame = num_end
        if num_full:
            self.td_num_full = num_full
        if randomize:
            self.td_rand_file = randomize
        if pgn_folder:
            self.td_pgn_folder = pgn_folder
        if end_length:
            self.td_end_length = end_length
        if full_length:
            self.td_full_length = full_length
        if batch_size:
            self.td_batch_size = batch_size

    def reset_adagrad(self):
        self.td_adagrad_acc = []
        for weight_shape in self.weight_shapes:
            self.td_adagrad_acc.append(np.zeros(weight_shape))

    def train_td(self, endgame, game_indices=None, start_idx=0, sts_scores=None):
        """
        Trains the neural net using TD-Leaf.
            Inputs:
                endgame [Boolean]
                    If True then only trains on endgames. Otherwise trains on a random subset of full games.
                game_indices [List of Ints]
                    Used when resuming. Provides (shuffled) list of indices for the games to train on.
                start_idx [Int]
                    Index of game_indices where training should be resumed.
        """

        assert not ((game_indices is None) and start_idx > 0)

        num_games = self.td_num_endgame if endgame else self.td_num_full

        # Only load some files if not random
        if self.td_rand_file:
            pgn_files = [f for f in os.listdir(self.td_pgn_folder) if
                         os.path.isfile(os.path.join(self.td_pgn_folder, f))]
        else:
            pgn_files = [f for f in os.listdir(self.td_pgn_folder)[:num_games] if
                         os.path.isfile(os.path.join(self.td_pgn_folder, f))]

        if game_indices is None:
            game_indices = range(num_games if num_games >= 0 else len(pgn_files))

            # Shuffle if necessary
            if self.td_rand_file:
                random.shuffle(game_indices)

        # Initialize STS scores if necessary
        if sts_scores is None and self.sts_on:
            sts_scores = []

        self.td_fen_index = 0
        if self.nn.training_mode == 'adagrad':
            self.reset_adagrad()

        for i in xrange(start_idx, len(game_indices)):

            game_idx = game_indices[i]
            fens = []

            # Open and use pgn file sequentially or at random
            with open(os.path.join(self.td_pgn_folder, pgn_files[game_idx])) as pgn:
                game = chess.pgn.read_game(pgn)

                if endgame:
                    # Only get endgame fens
                    curr_node = game.end()
                    for _ in range(self.td_end_length):
                        fens.insert(0, curr_node.board().fen())

                        # Check if start of game is reached
                        if curr_node == game.root():
                            break
                        curr_node = curr_node.parent
                else:
                    # Get all fens
                    curr_node = game.root()
                    while True:
                        fens.append(curr_node.board().fen())

                        # Check if end has been reached
                        if curr_node.is_end():
                            break

                        curr_node = curr_node.variations[0]

                    # Get random subset if working with a max size
                    if self.td_full_length != -1:
                        game_length = len(fens)
                        sub_start = random.randint(0, max(0, game_length - self.td_full_length))
                        sub_end = min(game_length, sub_start + self.td_full_length)
                        fens = fens[sub_start:sub_end]

                        # TODO: Remove this check later.
                        if (len(fens) != self.td_full_length and game_length >= self.td_full_length) or \
                                (len(fens) != game_length and game_length < self.td_full_length):
                            if self.verbose:
                                print "Warning: This shouldn't happen!"

            # Call TD-Leaf
            if self.verbose:
                print "Training on game %d of %d..." % (i + 1, num_games)
            self.td_leaf(fens)

            # Evaluate on STS if necessary
            if self.sts_on and ((i + 1) % self.sts_interval == 0):
                original_depth = self.guerilla.search.max_depth
                self.guerilla.search.max_depth = self.sts_depth
                sts_scores.append(Teacher.eval_sts(self.guerilla, mode=self.sts_mode)[0])
                self.guerilla.search.max_depth = original_depth
                if self.verbose:
                    print "STS Result: %s" % str(sts_scores[-1])

            # Check if out of time
            if self.out_of_time() and i != (len(game_indices) - 1):
                if self.verbose:
                    print "TD-Leaf " + ("endgame" if endgame else "fullgame") + " Timeout: Saving state and quitting."
                save = {'game_indices': game_indices,
                        'start_idx': i + 1,
                        'sts_scores': sts_scores}
                self.save_state(save)
                return

        if self.sts_on:
            self.print_sts(sts_scores)

        return

    def td_update_weights(self):
        """
        From batch of weight updates (gradients * discout values), find average. Updates adagrad accumulator.
        Updates weights.
        Note: The plurality of gradients is a function of the number nodes not the number of actual gradients.
        """
        avg_gradients = [grad / self.td_batch_size for grad in self.td_w_update]
        if self.nn.training_mode == 'adagrad':
            self.td_adagrad_acc = map(add, self.td_adagrad_acc, [grad ** 2 for grad in avg_gradients])
            learning_rates = [hp['TD_LRN_RATE'] / (np.sqrt(grad) + 1.0e-8) for grad in self.td_adagrad_acc]
        elif self.nn.training_mode == 'adadelta':
            raise NotImplementedError("TD leaf adadelta training has not yet been implemented")
        elif self.nn.training_mode == 'gradient_descent':
            learning_rates = [hp['TD_LRN_RATE']] * len(avg_gradients)
        else:
            raise NotImplementedError("Unrecognized training type")

        self.nn.add_all_weights([learning_rates[i] * avg_gradients[i] for i in xrange(len(avg_gradients))])

    def td_leaf(self, game):
        """
        Trains neural net using TD-Leaf algorithm.
            Inputs:
                Game [List]
                    A game consists of a sequential list of board states. Each board state is a FEN.
        """

        num_boards = len(game)
        game_info = [{'value': None, 'gradient': None} for _ in range(num_boards)]  # Indexed the same as num_boards

        # turn pruning for search off
        self.guerilla.search.reci_prune = False

        # Pre-calculate leaf value (J_d(x,w)) of search applied to each board
        # Get new board state from leaf
        # print "Calculating TD-Leaf values for move ",
        for i, board in enumerate(game):
            # print str(i) + "... ",
            value, _, board_fen = self.guerilla.search.run(chess.Board(board))

            # Get values and gradients for white plays next
            if dh.white_is_next(board_fen):
                game_info[i]['value'] = value
                game_info[i]['gradient'] = self.nn.get_all_weights_gradient(board_fen)
            else:
                # value is probability of WHITE winning
                game_info[i]['value'] = 1 - value

                # Get NEGATIVE gradient of a flipped board
                # Explanation:
                #   Flipped board = Black -> White, so now white plays next (as required by NN)
                #   Gradient of flipped board = Gradient of what used to be black
                #   Desired gradient = Gradient of what was originally white = - Gradient of flipped board
                game_info[i]['gradient'] = [-x for x in self.nn.get_all_weights_gradient(dh.flip_board(board_fen))]

        # turn pruning for search back on
        self.guerilla.search.reci_prune = True

        for t in range(num_boards):
            td_val = 0
            for j in range(t, num_boards - 1):
                # Calculate temporal difference
                dt = game_info[j + 1]['value'] - game_info[j]['value']
                # print dt
                # Add to sum
                td_val += math.pow(hp['TD_DISCOUNT'], j - t) * dt

            # Use gradient to update sum
            if not self.td_w_update:
                self.td_w_update = [w * td_val for w in game_info[t]['gradient']]
            else:
                # update each set of weights
                for i in range(len(game_info[t]['gradient'])):
                    self.td_w_update[i] += game_info[t]['gradient'][i] * td_val

            self.td_fen_index += 1

            if self.td_fen_index == self.td_batch_size:
                self.td_update_weights()
                self.td_fen_index = 0
                self.td_w_update = None

    # ---------- SELF-PLAY TRAINING METHODS

    def set_sp_params(self, num_selfplay=None, max_length=None):
        """
        Set the parameteres for self-play.
            Inputs:
                num_selfplay [Int]
                    Number of games to play against itself and train using TD-Leaf.
                max_length [Int]
                    Maximum number of moves in each self-play. (-1 = No max)
        """

        if num_selfplay:
            self.sp_num = num_selfplay
        if max_length:
            self.sp_length = max_length

    def train_selfplay(self, game_indices=None, start_idx=0, sts_scores=None):
        """
        Trains neural net using TD-Leaf algorithm based on partial games which the neural net plays against itself.
        Self-play is performed from a random board position. The random board position is found by loading from the fens
        file and then applying a random legal move to the board.
        """

        fens = cgp.load_fens(num_values=self.sp_num)

        if game_indices is None:
            game_indices = np.random.choice(len(fens), self.sp_num)

        max_len = float("inf") if self.sp_length == -1 else self.sp_length

        # Initialize STS scores if necessary
        if sts_scores is None and self.sts_on:
            sts_scores = []

        self.td_fen_index = 0
        if self.nn.training_mode == 'adagrad':
            self.reset_adagrad()

        for i in xrange(start_idx, len(game_indices)):
            if self.verbose:
                print "Generating self-play game %d of %d..." % (i + 1, self.sp_num)
            # Load random fen
            board = chess.Board(
                fens[game_indices[i]])  # white plays next, turn counter & castling unimportant here

            # Play random move to increase game variability
            board.push(random.sample(board.legal_moves, 1)[0])

            # Play a game against yourself
            game_fens = [board.fen()]
            for _ in range(max_len):
                # Check if game finished
                if board.is_checkmate():
                    break

                # Play move
                try:
                    board.push(self.guerilla.get_move(board))
                except AttributeError:
                    # TODO: Remove once bug is fixed
                    print board.fen()
                    raise

                # Store fen
                game_fens.append(board.fen())

            # Send game for TD-leaf training
            if self.verbose:
                print "Training on game %d of %d..." % (i + 1, self.sp_num)
            self.td_leaf(game_fens)

            # Evaluate on STS if necessary
            if self.sts_on and ((i + 1) % self.sts_interval == 0):
                original_depth = self.guerilla.search.max_depth
                self.guerilla.search.max_depth = self.sts_depth
                sts_scores.append(Teacher.eval_sts(self.guerilla, mode=self.sts_mode)[0])
                self.guerilla.search.max_depth = original_depth
                print "STS Result: %s" % str(sts_scores[-1])

            # Check if out of time
            if self.out_of_time() and i != (len(game_indices) - 1):
                if self.verbose:
                    print "TD-Leaf self-play Timeout: Saving state and quitting."
                save = {'game_indices': game_indices,
                        'start_idx': i + 1,
                        'sts_scores': sts_scores}
                self.save_state(save)
                return

        if self.sts_on:
            self.print_sts(sts_scores)

        return

    # ---------- STS EVALUATION

    sts_strat_files = ['activity_of_king', 'advancement_of_abc_pawns', 'advancement_of_fgh_pawns', 'bishop_vs_knight',
                       'center_control', 'knight_outposts', 'offer_of_simplification', 'open_files_and_diagonals',
                       'pawn_play_in_the_center', 'queens_and_rooks_to_the_7th_rank',
                       'recapturing', 'simplification', 'square_vacancy', 'undermining']
    sts_piece_files = ['pawn', 'bishop', 'rook', 'knight', 'queen', 'king']

    @staticmethod
    def eval_sts(player, mode="strategy"):
        """
        Evaluates the given player using the strategic test suite. Returns a score and a maximum score.
            Inputs:
                player [Player]
                    Player to be tested.
                mode [List of Strings] or [String]
                    Select the test mode(s), see below for options. By default runs "strategy".
                        "strategy": runs all strategic tests
                        "pieces" : runs all piece tests
                        other: specific EPD file
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
        sts_dir = os.path.dirname(os.path.abspath(__file__)) + '/../helpers/STS/'
        board = chess.Board()
        scores = []
        max_scores = []

        # Run tests
        for test in mode:
            print "Running %s STS test." % test
            # load STS epds
            epds = []
            if test == 'strategy':
                for filename in Teacher.sts_strat_files:
                    epds += Teacher.get_epds(sts_dir + filename + '.epd')
            elif test == 'sts_piece_files':
                for filename in Teacher.sts_piece_files:
                    epds += Teacher.get_epds(sts_dir + filename + '.epd')
            else:
                # Specific file
                try:
                    epds += Teacher.get_epds(sts_dir + test + '.epd')
                except IOError:
                    raise ValueError("Error %s is an invalid test mode." % test)

            # Test epds
            score = 0
            max_score = 0
            length = len(epds)
            print "STS: Scoring %s EPDS. Progress: " % length,
            print_perc = 5  # percent to print at
            for i, epd in enumerate(epds):
                # Print info
                if (i % (length / (100.0 / print_perc)) - 100.0 / length) < 0:
                    print "%d%% " % (i / (length / 100.0)),

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

                # Get move
                move = player.get_move(board)

                # score
                max_score += move_scores[ops['bm'][0]]
                try:
                    score += move_scores[move]
                except KeyError:
                    # Score of 0
                    pass
            print ""

            # append
            scores.append(score)
            max_scores.append(max_score)

        return scores, max_scores

    @staticmethod
    def get_epds(filename):
        """
        Returns a list of epds from the given file.
        Input:
            filename [String]
                Filename of EPD file to open. Must include absolute path.
        Output:
            epds [List of Strings]
                List of epds.
        """
        f = open(filename)
        epds = [line.rstrip() for line in f]
        f.close()

        return epds

    def print_sts(self, scores):
        """ Prints the STS scores and corresponding intervals. """

        if len(scores[0]) == 1:
            score_out = [score[0] for score in scores]
        else:
            score_out = scores

        print "Intervals: " + ",".join(map(str, [(x + 1) * self.sts_interval for x in range(len(scores))]))
        print "Scores: " + ",".join(map(str, score_out))


def direction_test():
    with Guerilla('Harambe', 'w', _load_file='weights_train_bootstrap_20160930-193556.p') as g:
        g.search.max_depth = 0
        t = Teacher(g)
        board = chess.Board()

        # num_vars
        num_fen = 2
        num_td = 100

        # Build fens
        fens = [None] * num_fen
        for i in range(num_fen):
            fens[i] = board.fen()
            board.push(list(board.legal_moves)[0])

        # print initial evaluations
        for i in range(num_td):
            curr_vals = []
            for j in range(num_fen / 2):
                curr_vals.append(g.nn.evaluate(fens[2 * j]))
                curr_vals.append(1 - g.nn.evaluate(dh.flip_board(fens[2 * j + 1])))

            a = curr_vals[0]
            b = curr_vals[1]
            print "%d,%f,%f,%f" % (i, a, b, abs(b - a))

            # run td leaf
            t.td_leaf(fens)


def main():

    run_time = 0
    if len(sys.argv) >= 2:
        hours = int(sys.argv[1])
        run_time += hours * 3600
    if len(sys.argv) >= 3:
        minutes = int(sys.argv[2])
        run_time += minutes * 60
    if len(sys.argv) >= 4:
        seconds = int(sys.argv[3])
        run_time += seconds

    print "Training for %f hours" % (float(run_time)/3600.0)

    if run_time == 0:
        run_time = None

    with Guerilla('Harambe', 'w', training_mode='adagrad') as g:
        g.search.max_depth = 1
        t = Teacher(g)
        t.set_bootstrap_params(num_bootstrap=50000)  # 488037
        t.set_td_params(num_end=5, num_full=12, randomize=False, end_length=10, full_length=12)
        t.set_sp_params(num_selfplay=10, max_length=12)
        t.sts_on = False
        t.sts_interval = 100
        # t.sts_mode = Teacher.sts_strat_files[0]
        t.run(['train_bootstrap'], training_time=run_time)
        # t.run(['load_and_resume'], training_time=28000)


if __name__ == '__main__':
    main()
