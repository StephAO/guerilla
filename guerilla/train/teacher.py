import math
import os
import pickle
import random
import sys
import time
from operator import add

import chess
import chess.pgn
import matplotlib.pyplot as plt
import numpy as np
import yaml
from pkg_resources import resource_filename

import guerilla.data_handler as dh
import guerilla.train.chess_game_parser as cgp
import guerilla.train.stockfish_eval as sf
from guerilla.players import Guerilla
from guerilla.train.sts import eval_sts, sts_strat_files


class Teacher:
    actions = [
        'train_bootstrap',
        'train_td_end',
        'train_td_full',
        'train_selfplay',
        'load_and_resume'
    ]

    def __init__(self, _guerilla, hp_load_file=None, training_mode=None, 
                 test=False, verbose=True, **hp):
        """
            Initialize teacher, sets member variables.

            Inputs:
                _guerilla [Guerilla]:
                    Guerilla to train
                training_mode [String]:
                    Training mode to be used. Defaults to Adagrad.
                test [Bool]:
                    Set to true if its a test. If true, doesn't save weights.
                verbose [Bool]:
                    If true, teacher prints more output.
                **hp[**kwargs]:
                    Hyper parameters in keyword format. Keyword must match hyper
                    parameter name. See self._set_hyper_params for valid hyper
                    parameters. Hyper parameters defined in hp will overwrite
                    params loaded from a file.
        """
        # dictionary of different training/evaluation methods

        self.guerilla = _guerilla
        self.nn = _guerilla.nn
        self.start_time = None
        self.training_time = None
        self.prev_checkpoint = None
        self.checkpoint_interval = None  # Checkpoint interval (seconds). 'None' -> only saves on timeout/completion
        self.actions = None
        self.curr_action_idx = None
        self.saved = None

        self.test = test
        self.verbose = verbose

        # Bootstrap parameters
        self.num_bootstrap = -1
        self.conv_loss_thresh = 0.0001  # 
        self.conv_window_size = 20  # Number of epochs to consider when checking for convergence
        self.keep_prob = 1.0 # Probability of dropout during neural network training

        # TD-Leaf parameters
        self.td_pgn_folder = resource_filename('guerilla', 'data/pgn_files/single_game_pgns')
        self.td_rand_file = False  # If true then TD-Leaf randomizes across the files in the folder.
        self.td_num_endgame = -1  # The number of endgames to train on using TD-Leaf (-1 = All)
        self.td_num_full = -1  # The number of full games to train on using TD-Leaf
        self.td_end_length = 12  # How many moves are included in endgame training
        self.td_full_length = -1  # Maximum number of moves for full game training (-1 = All)
        self.td_w_update = None
        self.td_fen_index = 0
        self.td_batch_size = 50

        self.hp = {}
        if hp_load_file is None:
            hp_load_file = 'default.yaml'
        self.set_hyper_params_from_file(hp_load_file)
        self.set_hyper_params(**hp)

        if training_mode is None:
            training_mode = 'adagrad'
        self.training_mode = training_mode
        self.train_step = self.nn.init_training(self.training_mode, learning_rate = self.hp['LEARNING_RATE'],
                                                reg_const = self.hp['REGULARIZATION_CONST'], loss_fn = self.nn.MSE,
                                                decay_rate= self.hp['DECAY_RATE'])

        if self.verbose:
            print "Training neural net using %s." % self.training_mode

        if self.training_mode == 'adagrad':
            self.td_adagrad_acc = None
            self.weight_shapes = []
            for weight in self.nn.all_weights_biases:
                self.weight_shapes.append(weight.get_shape().as_list())

        # Self-play parameters
        self.sp_num = 1  # The number of games to play against itself
        self.sp_length = 12  # How many moves are included in game playing

        # STS Evaluation Parameters
        self.sts_on = False  # Whether STS evaluation occurs during training
        self.sts_interval = 50  # Interval at which STS evaluation occurs, unit is number of games
        self.sts_mode = "strategy"  # STS mode for evaluation
        self.sts_depth = self.guerilla.search.max_depth  # Depth used for STS evaluation (can override default)

        # Build unique file modifier which demarks final output files from this session
        self.file_modifier = "_%s%s%sFC.p" % (time.strftime("%m%d-%H%M"), 
                                '_conv' if self.guerilla.nn.hp['USE_CONV'] else '',
                                '_' + str(self.nn.hp['NUM_FC']))


    # ---------- RUNNING AND RESUMING METHODS

    def run(self, actions, training_time=None):
        """
            1. load data from file
            2. configure data
            3. run actions
        """
        self.start_time = self.prev_checkpoint = time.time()
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
                weight_file = "w_" + self.actions[self.curr_action_idx - 1] + self.file_modifier
                self.nn.save_weight_values(_filename=weight_file)

            # Check if done
            if self.curr_action_idx >= len(self.actions):
                break
            elif self.out_of_time():
                # Check if run out of time

                if not self.saved:
                    # Ran out of time between actions, save data
                    if self.verbose:
                        print "State saved between actions."
                    self.save_state(state={})
                else:
                    # Ran out of time during an action.
                    # Don't save data, but note that still on previous action.
                    if self.verbose:
                        print "State saved during an action."
                break
            elif self.checkpoint_reached():
                # A checkpoint as been reached!
                self.save_state(state={}, is_checkpoint=True)

            # Else
            action = self.actions[self.curr_action_idx]
            if action == 'train_bootstrap':
                if self.verbose:
                    print "Performing Bootstrap training!"
                    print "Fetching stockfish values..."

                fens = cgp.load_fens(num_values=self.num_bootstrap)
                if (len(fens) % self.hp['BATCH_SIZE']) != 0:
                    fens = fens[:(-1) * (len(fens) % self.hp['BATCH_SIZE'])]
                true_values = sf.load_stockfish_values(num_values=len(fens))

                self.train_bootstrap(fens, true_values)
            elif action == 'train_td_end':
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

            if not self.saved:
                # If not timed out
                self.curr_action_idx += 1

    def set_hyper_params_from_file(self, file):
        """
            Updates hyper parameters from a yaml file.
            Will only affect hyper parameters that are provided. Unspecified
            hyper parameters will not change.
            Inputs:
                file [String]:
                    filename to use. File must be in data/hyper_params/teacher/
        """
        relative_filepath = 'data/hyper_params/teacher/' + file
        filepath = resource_filename('guerilla', relative_filepath)
        with open(filepath, 'r') as yaml_file:
            self.hp.update(yaml.load(yaml_file))

    def set_hyper_params(self, **hyper_parameters):
        """
            Updates hyper parameters from arguments.
            Will only affect hyper parameters that are provided. Unspecified
            hyper parameters will not change.
            Hyper parameters that are used:
                "NUM_EPOCHS" - number of epochs to train on
                "BATCH_SIZE" - number of positions used for a single weight update
                "LEARNING_RATE" - constant used to scale the gradient before the
                                  value is used to update weights in bootstrap
                "VALIDATION_SIZE" - Number of positions used to calculate
                                    validation loss.
                                    None of the positions in validation set are
                                    used in training. 
                                    VALIDATION SIZE % BATCH SIZE == 0 
                                    VALIDATION SIZE + BATCH SIZE < Number of fens provided
                "TRAIN_CHECK_SIZE" - Number of positions used to calculate
                                     training loss.
                                     All of the positions in validation set are
                                     used in training.
                "LOSS_THRESHOLD" - If all loss changes in the window <= 
                                   than LOSS_THRESHOLD then weights have converged
                "TD_LRN_RATE" - constant used to scale the gradient before the
                                value is used to update weights in td_leaf
                "TD_DISCOUNT" - Discount rate used by td leaf
                                i.e. The next board has TD_DISCOUNT amount of
                                weight compared to current board in regards to
                                temporal difference value.
                "REGULARIZATION_CONST" - Constant used to determine value of
                                         regularization term
                "DECAY_RATE" -Used in AdaDelta
            Inputs:
                hyperparmeters [**kwargs]:
                    hyperparameters to update with
        """
        self.hp.update(hyper_parameters)

    def save_state(self, state, filename="state.p", is_checkpoint=False):
        """
            Save current state so that training can resume at another time
            Note: Can only save state at whole batch intervals (but mid-epoch is fine)
            Inputs:
                state [dict]:
                    For bootstrap:
                        game_indices[list of ints]
                        (if action == 'train_bootstrap')
                            epoch_num[int]
                            loss[list of floats]
                        (if action == 'train_td_...')
                            start_idx [int]
                is_checkpoint [Boolean]:
                    Denotes if the save is a checkpoint (True) or a timeout save (False).
                filename [string]:
                    filename to save pickle to
        """

        if is_checkpoint:
            self.prev_checkpoint = time.time()
            if self.verbose:
                print "Checkpoint reached at " + time.strftime('%Y-%m-%d %H:%M:%S',
                                                               time.localtime(self.prev_checkpoint))

        state['curr_action_idx'] = self.curr_action_idx
        state['actions'] = self.actions
        state['prev_checkpoint'] = self.prev_checkpoint
        state['save_time'] = self.prev_checkpoint if is_checkpoint else time.time()

        # Save training parameters
        state['td_leaf_param'] = {'randomize': self.td_rand_file,
                                  'num_end': self.td_num_endgame,
                                  'num_full': self.td_num_full,
                                  'end_length': self.td_end_length,
                                  'full_length': self.td_full_length,
                                  'batch_size': self.td_batch_size,
                                  }
        if self.training_mode == 'adagrad':
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
        train_var_path = resource_filename('guerilla', 'data/train_checkpoint/in_training_vars.vars')
        train_var_file = self.nn.save_training_vars(train_var_path)
        if train_var_file:
            state['train_var_file'] = train_var_file

        pickle_path = resource_filename('guerilla', 'data/train_checkpoint/' + filename)
        self.nn.save_weight_values(_filename='in_training_weight_values.p')
        pickle.dump(state, open(pickle_path, 'wb'))
        self.saved = not is_checkpoint

    def load_state(self, filename='state.p'):
        """
            Load state from pickle. Returns information necessary to resume training
            Inputs:
                filename[string]:
                    filename to load pickle from
            Outputs:
                state [dict]:
                    action [string]
                    epoch_num [int]
                    remaining_boards [list of ints]
                    loss [list of floats]
        """
        pickle_path = resource_filename('guerilla', 'data/train_checkpoint/' + filename)
        state = pickle.load(open(pickle_path, 'rb'))

        # Update checkpoint time
        self.prev_checkpoint = time.time() - (state['save_time'] - state['prev_checkpoint'])

        # Load training parameters
        self.set_td_params(**state.pop('td_leaf_param'))
        self.set_sp_params(**state.pop('sp_param'))

        # Load adagrad params
        if self.training_mode == 'adagrad':
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
            if (len(fens) % self.hp['BATCH_SIZE']) != 0:
                fens = fens[:(-1) * (len(fens) % self.hp['BATCH_SIZE'])]

            true_values = sf.load_stockfish_values(num_values=len(fens))

            # finish epoch
            train_fens = fens[:(-1) * self.hp['VALIDATION_SIZE']]  # fens to train on
            train_values = true_values[:(-1) * self.hp['VALIDATION_SIZE']]
            self.weight_update_bootstrap(train_fens, train_values, state['game_indices'], self.train_step)

            # Continue with rest of epochs
            self.train_bootstrap(fens, true_values, start_epoch=state['epoch_num'] + 1,
                                 loss=state['loss'], train_loss=state['train_loss'])
        elif action == 'train_td_end':
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
            raise ValueError("Error: Trying to resume on a resume call - This shouldn't happen.")
        else:
            raise NotImplementedError("Error: %s is not a valid action." % action)

        self.curr_action_idx += 1
        return

    def out_of_time(self):
        """
        Returns True if training has run out of time. False otherwise.
            Output:
                [Boolean]
        """
        return self.training_time is not None and time.time() - self.start_time >= self.training_time

    def checkpoint_reached(self):
        """
        Returns True if a checkpoint has been reached. False otherwise.
            Output:
                [Boolean]
        """
        return self.checkpoint_interval is not None and time.time() - self.prev_checkpoint >= self.checkpoint_interval

    # ---------- BOOTSTRAP TRAINING METHODS
    def set_bootstrap_params(self, num_bootstrap=None):
        self.num_bootstrap = num_bootstrap

    def train_bootstrap(self, fens, true_values, start_epoch=0, loss=None, train_loss=None):
        """
            Train neural net

            Inputs:
                fens [list of strings]:
                    The fens representing the board states
                true_values [ndarray]:
                    Expected output for each chess board state (between 0 and 1)
        """

        train_check_spacing = (len(fens) - self.hp['VALIDATION_SIZE']) / self.hp['TRAIN_CHECK_SIZE']

        train_fens = fens[:(-1) * self.hp['VALIDATION_SIZE']]  # fens to train on
        valid_fens = fens[(-1) * self.hp['VALIDATION_SIZE']:]  # fens to check convergence on
        train_check_fens = train_fens[::train_check_spacing]  # fens to evaluate training error on

        train_values = true_values[:(-1) * self.hp['VALIDATION_SIZE']]
        valid_values = true_values[(-1) * self.hp['VALIDATION_SIZE']:]
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
            print "%16d %16.5f %16.5f" % (start_epoch, loss[-1], train_loss[-1])
        for epoch in xrange(start_epoch, self.hp['NUM_EPOCHS']):
            if self.is_converged(loss):
                if self.verbose:
                    print "Training complete: Reached convergence threshold"
                break

            # Configure data (shuffle fens -> fens to channel -> group batches)
            game_indices = range(num_boards)
            random.shuffle(game_indices)

            # update weights
            timeout, state = self.weight_update_bootstrap(train_fens, train_values, game_indices, self.train_step)

            # save state if timeout or checkpoint
            if timeout or self.checkpoint_reached():
                state['epoch_num'] = epoch
                state['loss'] = loss
                state['train_loss'] = train_loss

                # Timeout
                if timeout:
                    self.save_state(state)
                    return

                # Checkpoint
                state['game_indices'] = []
                self.save_state(state, is_checkpoint=True)

            # evaluate nn for convergence
            loss.append(self.evaluate_bootstrap(valid_fens, valid_values))
            train_loss.append(self.evaluate_bootstrap(train_check_fens, train_check_values))
            if self.verbose:
                print "%16d %16.5f %16.5f" % (epoch + 1, loss[-1], train_loss[-1])

        else:
            if self.verbose:
                print "Training complete: Reached max epoch, no convergence yet"

        if self.test:
            filename = resource_filename('guerilla', 'data/loss/loss_test.p')
        else:
            filename = resource_filename('guerilla', 'data/loss/loss' + self.file_modifier)

            # save loss
        pickle.dump({"loss": loss, "train_loss": train_loss},
                    open(filename, 'wb'))
        # plt.plot(range(epoch + 1), error)
        # plt.show()

        return

    def weight_update_bootstrap(self, fens, true_values_, game_indices, train_step):
        """ Weight update for multiple batches"""

        if len(game_indices) % self.hp['BATCH_SIZE'] != 0:
            raise Exception("Error: number of fens provided (%d) is not a multiple of batch_size (%d)" %
                            (len(game_indices), self.hp['BATCH_SIZE']))

        num_batches = int(len(game_indices) / self.hp['BATCH_SIZE'])

        board_num = 0
        if self.nn.hp['NN_INPUT_TYPE'] == 'giraffe':
            boards = np.zeros((self.hp['BATCH_SIZE'], dh.GF_FULL_SIZE))
        elif self.nn.hp['NN_INPUT_TYPE'] == 'bitmap':
            boards = np.zeros((self.hp['BATCH_SIZE'], 8, 8, self.nn.hp['NUM_CHANNELS']))
            if self.nn.hp['USE_CONV']:
                diagonals = np.zeros((self.hp['BATCH_SIZE'], 10, 8, self.nn.hp['NUM_CHANNELS']))
        true_values = np.zeros(self.hp['BATCH_SIZE'])

        for i in xrange(num_batches):
            # if training time is up, save state
            if self.out_of_time():
                if self.verbose:
                    print "Bootstrap Timeout: Saving state and quitting"
                return True, {'game_indices': game_indices[(i * self.hp['BATCH_SIZE']):]}

            # set up batch
            for j in xrange(self.hp['BATCH_SIZE']):
                boards[j] = dh.fen_to_nn_input(fens[game_indices[board_num]], 
                                               self.nn.hp['NN_INPUT_TYPE'],
                                               self.nn.hp['NUM_CHANNELS'])
                if self.nn.hp['NN_INPUT_TYPE'] == 'bitmap' and self.nn.hp['USE_CONV']:
                    diagonals[j] = dh.get_diagonals(boards[j], self.nn.hp['NUM_CHANNELS'])
                true_values[j] = true_values_[game_indices[board_num]]
                board_num += 1

            _feed_dict = {self.nn.data: boards, self.nn.true_value: true_values, self.nn.keep_prob: self.keep_prob}
            if self.nn.hp['NN_INPUT_TYPE'] == 'bitmap' and self.nn.hp['USE_CONV']:
                _feed_dict[self.nn.data_diags] = diagonals
            # train batch
            self.nn.sess.run([train_step], feed_dict=_feed_dict)

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

        if len(fens) % self.hp['BATCH_SIZE'] != 0:
            raise Exception("Error: Validation set size (%d) is not a multiple of batch_size (%d)" %
                            (len(fens), self.hp['BATCH_SIZE']))

        # Configure data
        num_batches = int(len(fens) / self.hp['BATCH_SIZE'])

        board_num = 0

        if self.nn.hp['NN_INPUT_TYPE'] == 'giraffe':
            # Configure data
            boards = np.zeros((self.hp['BATCH_SIZE'], dh.GF_FULL_SIZE))

        elif self.nn.hp['NN_INPUT_TYPE'] == 'bitmap':
            # Configure data
            boards = np.zeros((self.hp['BATCH_SIZE'], 8, 8, self.nn.hp['NUM_CHANNELS']))
            if self.nn.hp['USE_CONV']:
                diagonals = np.zeros((self.hp['BATCH_SIZE'], 10, 8, self.nn.hp['NUM_CHANNELS']))
        true_values_batch = np.zeros(self.hp['BATCH_SIZE'])

        # Initialize Error
        error = 0

        for i in xrange(num_batches):
            # set up batch
            for j in xrange(self.hp['BATCH_SIZE']):
                boards[j] = dh.fen_to_nn_input(fens[board_num], 
                                               self.nn.hp['NN_INPUT_TYPE'],
                                               self.nn.hp['NUM_CHANNELS'])
                if self.nn.hp['NN_INPUT_TYPE'] == 'bitmap' and self.nn.hp['USE_CONV']:
                    diagonals[j] = dh.get_diagonals(boards[j], self.nn.hp['NUM_CHANNELS'])
                true_values_batch[j] = true_values[board_num]
                board_num += 1

            _feed_dict = {self.nn.data: boards, self.nn.true_value: true_values_batch}
            if self.nn.hp['NN_INPUT_TYPE'] == 'bitmap' and self.nn.hp['USE_CONV']:
                _feed_dict[self.nn.data_diags] = diagonals

            # Get batch loss
            error += self.nn.sess.run(self.nn.MSE, feed_dict=_feed_dict)

        return error

    def is_converged(self, loss):
        """
        Checks if the loss has converged.
        Input:
            loss [List of Floats]
                The current loss
        Output:
            [Boolean]
                True if the loss has converged, False otherwise.
        """

        # +1 because we are looking at the number of changes
        if len(loss) < (self.conv_window_size + 1):
            return False

        # Check if any items in the window indicate non-convergence
        for i in range(1, self.conv_window_size + 1):
            if abs(loss[- (i + 1)] - loss[-i]) > self.hp['LOSS_THRESHOLD']:
                return False

        return True

    # ---------- TD-LEAF TRAINING METHODS

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
        if self.training_mode == 'adagrad':
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

            # Call TD-Leaf
            if self.verbose:
                print "Training on game %d of %d..." % (i + 1, num_games)
            self.td_leaf(fens)

            # Evaluate on STS if necessary
            if self.sts_on and ((i + 1) % self.sts_interval == 0):
                original_depth = self.guerilla.search.max_depth
                self.guerilla.search.max_depth = self.sts_depth
                sts_scores.append(eval_sts(self.guerilla, mode=self.sts_mode)[0])
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
        if self.training_mode == 'adagrad':
            self.td_adagrad_acc = map(add, self.td_adagrad_acc, [grad ** 2 for grad in avg_gradients])
            learning_rates = [self.hp['TD_LRN_RATE'] / (np.sqrt(grad) + 1.0e-8) for grad in self.td_adagrad_acc]
        elif self.training_mode == 'adadelta':
            raise NotImplementedError("TD leaf adadelta training has not yet been implemented")
        elif self.training_mode == 'gradient_descent':
            learning_rates = [self.hp['TD_LRN_RATE']] * len(avg_gradients)
        else:
            raise NotImplementedError("Unrecognized training type")

        self.nn.add_to_all_weights([learning_rates[i] * avg_gradients[i] for i in xrange(len(avg_gradients))])

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
                # Add to sum
                td_val += math.pow(self.hp['TD_DISCOUNT'], j - t) * dt

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

        # shuffle fens
        random.shuffle(fens)

        if game_indices is None:
            game_indices = np.random.choice(len(fens), self.sp_num)

        max_len = float("inf") if self.sp_length == -1 else self.sp_length

        # Initialize STS scores if necessary
        if sts_scores is None and self.sts_on:
            sts_scores = []

        self.td_fen_index = 0
        if self.training_mode == 'adagrad':
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
                board.push(self.guerilla.get_move(board))

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
                sts_scores.append(eval_sts(self.guerilla, mode=self.sts_mode)[0])
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

    def print_sts(self, scores):
        """ Prints the STS scores and corresponding intervals. """

        if len(scores[0]) == 1:
            score_out = [score[0] for score in scores]
        else:
            score_out = scores

        print "Intervals: " + ",".join(map(str, [(x + 1) * self.sts_interval for x in range(len(scores))]))
        print "Scores: " + ",".join(map(str, score_out))


def direction_test():
    with Guerilla('Harambe', 'w', load_file='weights_train_bootstrap_20160930-193556.p') as g:
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

    print "Training for %f hours" % (float(run_time) / 3600.0)

    if run_time == 0:
        run_time = None

    with Guerilla('Harambe', 'w') as g:
        g.search.max_depth = 1
        t = Teacher(g, training_mode='adagrad')
        t.set_bootstrap_params(num_bootstrap=1000)  # 488037
        t.set_td_params(num_end=5, num_full=12, randomize=False, end_length=2, full_length=12)
        t.set_sp_params(num_selfplay=10, max_length=12)
        t.sts_on = False
        t.sts_interval = 100
        t.checkpoint_interval = None
        t.run(['train_bootstrap'], training_time=run_time)
        print eval_sts(g)
        # t.run(['load_and_resume'], training_time=28000)


if __name__ == '__main__':
    main()
