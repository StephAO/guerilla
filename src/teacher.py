import os
import random
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
        }

        self.guerilla = _guerilla
        self.nn = _guerilla.nn
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.actions = actions

    def run(self):
        """ 
            1. load data from file
            2. configure data
            3. run actions
        """
        fens_filename = "fens.p"
        stockfish_filename = "sf_scores.p"

        fens = cgp.load_fens(fens_filename)

        num_batches = len(fens) / BATCH_SIZE

        true_values = sf.load_stockfish_values(stockfish_filename)
        true_values = np.reshape(true_values[:num_batches * BATCH_SIZE], (num_batches, BATCH_SIZE))

        print "Finished getting stockfish values. Begin training neural_net with %d items" % (len(fens))

        boards = np.zeros((num_batches, BATCH_SIZE, 8, 8, NUM_CHANNELS))
        diagonals = np.zeros((num_batches, BATCH_SIZE, 10, 8, NUM_CHANNELS))

        for action in self.actions:
            if action in self.actions_dict:
                self.actions_dict[action](boards, diagonals, true_values, num_batches, fens)
            else:
                print "Error: %s is not a valid command" % (action)

    def train_bootstrap(self, boards, diagonals, true_values, num_batches, fens, save_weights=True):
        """
            train neural net
        """

        # train should depend on action
        raw_input('This will overwrite your old weights\' pickle, do you still want to proceed? (Hit Enter)')
        print 'Training data. Will save weights to pickle'

        for epoch in xrange(NUM_EPOCHS):
            # Configure data (shuffle fens -> fens to channel -> group batches)
            game_indices = range(num_batches * BATCH_SIZE)
            random.shuffle(game_indices)
            for game_idx in game_indices:
                batch_num = game_idx / BATCH_SIZE
                batch_idx = game_idx % BATCH_SIZE

                boards[batch_num][batch_idx] = dc.fen_to_channels(fens[game_idx])
                for j in xrange(BATCH_SIZE):  # Maybe use matrices instead of for loop for speed
                    diagonals[batch_num][batch_idx] = dc.get_diagonals(boards[batch_num][batch_idx])
            # train epoch
            self.weight_update_bootstrap(boards, diagonals, true_values)

        # evaluate nn
        self.evaluate(boards, diagonals, true_values)

    # move save_weights to come from action
    def weight_update_bootstrap(self, boards, diagonals, true_values, save_weights=True):
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
                save_weights[Bool] (optional - default = True):
                    Save weights to file after training
        """

        # From my limited understanding x_entropy is not suitable - but if im wrong it could be better
        # Using squared error instead
        cost = tf.reduce_sum(tf.pow(tf.sub(self.nn.pred_value, self.nn.true_value), 2))

        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

        for i in xrange(np.shape(boards)[0]):
            self.nn.sess.run([train_step], feed_dict={self.nn.data: boards[i], self.nn.data_diags: diagonals[i],
                                                      self.nn.true_value: true_values[i]})

        if save_weights:
            self.nn.save_weight_values()

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