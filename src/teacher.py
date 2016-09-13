import os
import random
import time
import numpy as np
import tensorflow as tf

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
        true_values = sf.load_stockfish_values(stockfish_filename)[:len(fens)]

        self.start_time = time.time()
        self.training_time = training_time
        for action in self.actions:
            if action in self.actions_dict:
                    self.actions_dict[action](boards, diagonals, true_values, fens)
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
        pickle.dump(weight_values, open(pickle_path, 'wb'))
        self.nn.save_weight_values(filename = 'in_training_weight_values.p')

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
        state = self.load_state()
        fens = cgp.load_fens(self.files[0])

        num_batches = len(fens) / BATCH_SIZE

        true_values = sf.load_stockfish_values(self.files[0])[:len(fens)]

        self.start_time = time.time()
        self.training_time = training_time;

        if state['action'] == 'boostrap':
            # finish epoch
            cost = tf.reduce_sum(tf.pow(tf.sub(self.nn.pred_value, self.nn.true_value), 2))
            train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
            self.weight_update_bootstrap(fens, true_values, state['remaining_boards'], cost, train_step)
            # continue with rests of epochs
            self.train_bootstrap(boards, diagonals, true_values, fens, start_epoch=state['epoch_num'])

    def train_bootstrap(self, boards, diagonals, true_values, fens, start_epoch = 0):
        """
            Train neural net

            Inputs:
                nn[NeuralNet]:
                    Neural net to train
                boards[ndarray]:
                    Chess board states to train neural net on. Must in correct input
                    format - See fen_to_channels in main.py
                diagonals[ndarray]:
                    Diagonals of chess board states to train neural net on. Must in 
                    correct input format - See get_diagonals in main.py
                true_values[ndarray]:
                    Expected output for each chess board state (between 0 and 1)
                num_batches[int]:
                    number of batches
                fens[list of strings]:
                    the fens
        """
        raw_input('This will overwrite your old weights\' pickle, do you still want to proceed? (Hit Enter)')
        print 'Training data. Will save weights to pickle'

        num_boards = len(fens)

        # From my limited understanding x_entropy is not suitable - but if im wrong it could be better
        # Using squared error instead
        cost = tf.reduce_sum(tf.pow(tf.sub(self.nn.pred_value, self.nn.true_value), 2))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

        for epoch in xrange(start_epoch, NUM_EPOCHS):
            # Configure data (shuffle fens -> fens to channel -> group batches)
            game_indices = range(num_boards)
            random.shuffle(game_indices)

            save = self.weight_update_bootstrap(fens, true_values, game_indices, cost, train_step)

            if save[0]:
                save[1]['epoch_num'] = epoch + 1
                save_state(save[1])

        # evaluate nn
        self.evaluate(boards, diagonals, true_values)

    def weight_update_bootstrap(self, fens, true_values_, game_indices, cost, train_step):
        """ Weight update for a single batch """        

        if len(game_indices) % BATCH_SIZE != 0:
            raise Exception("Error: number of actual fens and expected fens do not match up")

        num_batches = int(len(game_indices) / BATCH_SIZE)

        board_num = 0
        boards = np.zeros((BATCH_SIZE, 8, 8, NUM_CHANNELS))
        diagonals = np.zeros((BATCH_SIZE, 10, 8, NUM_CHANNELS))
        true_values = np.zeros((BATCH_SIZE))

        for i in xrange(num_batches):
            # if training time is up, save state
            if self.training_time is not None 
                and time.time() - self.start_time >= self.training_time:
                return (True, 
                        {   
                            'remaining_boards'   : game_indices[i*BATCH_SIZE+1:]
                            'action'            : 'train_bootstrap'
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


    def evaluate(self, boards, diagonals, true_values):
        """
            Evaluate neural net

            Inputs:
                nn[NeuralNet]:
                    Neural net to evaluate
                boards[ndarray]:
                    Chess board states to evaluate neural net on. Must in correct 
                    input format - See fen_to_channels in main.py
                diagonals[ndarray]:
                    Diagonals of chess board states to evaluate neural net on. 
                    Must in correct input format - See get_diagonals in main.py
                true_values[ndarray]:
                    Expected output for each chess board state (between 0 and 1)
        """

        total_boards = 0
        right_boards = 0
        mean_error = 0

        pred_value = tf.reshape(self.nn.pred_value, [-1])
        err = tf.sub(self.nn.true_value, pred_value)
        err_sum = tf.reduce_sum(err)

        guess_whos_winning = tf.equal(tf.round(self.nn.true_value), tf.round(pred_value))
        num_right = tf.reduce_sum(tf.cast(guess_whos_winning, tf.float32))

        for i in xrange(np.shape(boards)[0]):
            es, nr, gww, pv = self.nn.sess.run([err_sum, num_right, guess_whos_winning, pred_value],
                                               feed_dict={self.nn.data: boards[i], self.nn.data_diags: diagonals[i],
                                                          self.nn.true_value: true_values[i]})
            total_boards += len(true_values[i])
            right_boards += nr
            mean_error += es

        mean_error = mean_error / total_boards
        print "mean_error: %f, guess who's winning correctly in %d out of %d games" % (
            mean_error, right_boards, total_boards)


def main():
    g = guerilla.Guerilla('Harambe')
    t = Teacher(g, ['train_bootstrap'])
    t.run()


if __name__ == '__main__':
    main()