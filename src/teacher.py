import os
import random
import time
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

import guerilla
import data_configuring as dc
import stockfish_eval as sf
import chess_game_parser as cgp
from hyper_parameters import *

class Teacher:
    def __init__(self, _guerilla, actions):

        # dictionary of different training/evaluation methods
        self.actions_dict = {
            'train_bootstrap' : self.train_bootstrap,
            'load_and_resume' : self.resume
        }

        self.guerilla = _guerilla
        self.nn = _guerilla.nn
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.actions = actions
        self.start_time = None
        self.training_time = None
        self.files = None

    def run(self, training_time = None, fens_filename = "fens.p", stockfish_filename = "sf_scores.p"):
        """ 
            1. load data from file
            2. configure data
            3. run actions
        """
        self.files = [fens_filename, stockfish_filename]

        fens = cgp.load_fens(fens_filename)
        fens = fens[:(-1) * (len(fens) % BATCH_SIZE)]
        true_values = sf.load_stockfish_values(stockfish_filename)[:len(fens)]

        self.start_time = time.time()
        self.training_time = training_time
        for action in self.actions:
            if action in self.actions_dict:
                    self.actions_dict[action](fens, true_values)
            else:
                print "Error: %s is not a valid command" % (action)

    def save_state(self, state, filename = "state.p"):
        """
            Save current state so that training can resume at another time
            Note: Can only save state at whole batch intervals (but mid-epoch is fine)
            Inputs:
                state[dict]:
                    action[string]
                    epoch_num[int]
                    remaining_boards[list of ints]
                    loss[list of floats]
                filename[string]:
                    filename to save pickle to
        """
        state['files'] = self.files
        pickle_path = self.dir_path + '/../pickles/' + filename
        self.nn.save_weight_values(filename = 'in_training_weight_values.p')
        pickle.dump(state, open(pickle_path, 'wb'))

    def load_state(self, filename = 'state.p'):
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

        self.files = state['files']
        self.nn.load_weight_values(filename = 'in_training_weight_values.p')
        return state

    def resume(self, training_time = None):
        """
            Resumes training from a previously paused training session
        """
        print "Resuming training"
        state = self.load_state()
        fens = cgp.load_fens(self.files[0])

        num_batches = len(fens) / BATCH_SIZE

        true_values = sf.load_stockfish_values(self.files[1])[:len(fens)]

        self.start_time = time.time()
        self.training_time = training_time;

        if state['action'] == 'train_bootstrap':
            # finish epoch
            train_fens = fens[:(-1) * CONV_CHECK_SIZE] # fens to train on
            convg_fens = fens[(-1) * CONV_CHECK_SIZE:] # fens to check convergence on

            train_values = true_values[:(-1) * CONV_CHECK_SIZE]
            convg_values = true_values[(-1) * CONV_CHECK_SIZE:]
            cost = tf.reduce_sum(tf.pow(tf.sub(self.nn.pred_value, self.nn.true_value), 2))
            train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
            self.weight_update_bootstrap(train_fens, train_values, state['remaining_boards'], cost, train_step)

            # evaluate nn for convergence
            state['error'].append(self.evaluate_bootstrap(convg_fens, convg_values))
            base_loss = state['error'][0] - state['error'][1]
            curr_loss = state['error'][-2] - state['error'][-1]
            if (abs(curr_loss / base_loss) < CONV_THRESHOLD):
                self.nn.save_weight_values()
                plt.plot(range(state['epoch_num']), state['error'])
                plt.show()
                return

            # continue with rests of epochs
            self.train_bootstrap(fens, true_values, start_epoch=state['epoch_num'], error=state['error'])

    def train_bootstrap(self, fens, true_values, start_epoch = 0, error = []):
        """
            Train neural net

            Inputs:
                fens[list of strings]:
                    The fens representing the board states
                true_values[ndarray]:
                    Expected output for each chess board state (between 0 and 1)
                num_batches[int]:
                    number of batches
        """

        train_fens = fens[:(-1) * CONV_CHECK_SIZE] # fens to train on
        convg_fens = fens[(-1) * CONV_CHECK_SIZE:] # fens to check convergence on

        train_values = true_values[:(-1) * CONV_CHECK_SIZE]
        convg_values = true_values[(-1) * CONV_CHECK_SIZE:]

        num_boards = len(train_fens)

        usr_in = raw_input("This will overwrite your old weights\' pickle, do you still want to proceed (y/n)?: ")
        if usr_in.lower() == 'n':
            return
        print "Training data on %d positions. Will save weights to pickle" % (num_boards)

        # From my limited understanding x_entropy is not suitable - but if im wrong it could be better
        # Using squared error instead
        cost = tf.reduce_sum(tf.pow(tf.sub(self.nn.pred_value, self.nn.true_value), 2))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
        epoch = start_epoch-1
        for epoch in xrange(start_epoch, NUM_EPOCHS):
            print epoch
            # Configure data (shuffle fens -> fens to channel -> group batches)
            game_indices = range(num_boards)
            random.shuffle(game_indices)

            # update weights
            save = self.weight_update_bootstrap(train_fens, train_values, game_indices, cost, train_step)

            # save state if timeout
            if save[0]:
                save[1]['epoch_num'] = epoch + 1
                save[1]['error'] = error
                self.save_state(save[1])
                return

            # evaluate nn for convergence
            error.append(self.evaluate_bootstrap(convg_fens, convg_values))
            print error[-1]
            if (len(error) > 2):
                base_loss = error[0] - error[1]
                curr_loss = error[-2] - error[-1]
                if (abs(curr_loss / base_loss) < CONV_THRESHOLD):
                    print "Training complete: Reached convergence threshold"
                    break

        else:
            print "Training complete: Reached max epoch, no convergence yet"

        self.nn.save_weight_values()

        plt.plot(range(epoch+1), error)
        plt.show()
        return

    def weight_update_bootstrap(self, fens, true_values_, game_indices, cost, train_step):
        """ Weight update for a single batch """        

        if len(game_indices) % BATCH_SIZE != 0:
            raise Exception("Error: number of fens provided (%d) is not a multiple of batch_size (%d)" % (len(game_indices), BATCH_SIZE))

        num_batches = int(len(game_indices) / BATCH_SIZE)

        board_num = 0
        boards = np.zeros((BATCH_SIZE, 8, 8, NUM_CHANNELS))
        diagonals = np.zeros((BATCH_SIZE, 10, 8, NUM_CHANNELS))
        true_values = np.zeros((BATCH_SIZE))

        for i in xrange(num_batches):
            # if training time is up, save state
            if self.training_time is not None \
                and time.time() - self.start_time >= self.training_time:
                print "Timeout: Saving state and quitting"
                return (True, 
                        {   
                            'remaining_boards'   : game_indices[(i * BATCH_SIZE):],
                            'action'             : 'train_bootstrap'
                        })

            # set up batch
            for j in xrange(BATCH_SIZE):
                boards[j] = dc.fen_to_channels(fens[game_indices[board_num]])
                diagonals[j] = dc.get_diagonals(boards[j])
                true_values[j] = true_values_[game_indices[board_num]]
                board_num += 1

            # train batch
            self.nn.sess.run([train_step], feed_dict={  self.nn.data: boards, self.nn.data_diags: diagonals,
                                                        self.nn.true_value: true_values })

        return (False, {})


    def evaluate_bootstrap(self, fens, true_values):
        """
            Evaluate neural net

            Inputs:
                fens[list of strings]:
                    The fens representing the board states
                true_values[ndarray]:
                    Expected output for each chess board state (between 0 and 1)
        """

        total_boards = 0
        right_boards = 0
        mean_error = 0

        # Create tensors
        pred_value = tf.reshape(self.nn.pred_value, [-1])
        err = tf.sub(self.nn.true_value, pred_value)
        err_sum = tf.reduce_sum(err)

        # Configure data
        boards = np.zeros((CONV_CHECK_SIZE, 8, 8, NUM_CHANNELS))
        diagonals = np.zeros((CONV_CHECK_SIZE, 10, 8, NUM_CHANNELS))
        for i in xrange(CONV_CHECK_SIZE):
            boards[i] = dc.fen_to_channels(fens[i])
            diagonals[i] = dc.get_diagonals(boards[i])

        # Get loss
        error = self.nn.sess.run([err_sum], feed_dict = { 
                self.nn.data: boards, 
                self.nn.data_diags: diagonals,
                self.nn.true_value: true_values 
        })

        return abs(error[0])

def main():
    g = guerilla.Guerilla('Harambe')
    t = Teacher(g, ['train_bootstrap'])
    # t.run(training_time = 3600, fens_filename = "fens_1000.p", stockfish_filename = "true_values_1000.p")
    t.resume()

if __name__ == '__main__':
    main()