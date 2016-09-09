import tensorflow as tf
import numpy as np
import pickle
import os
from hyper_parameters import *
import data_configuring as dc


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv5x5_grid(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')  # Pad or fit? (same is pad, fit is valid)


def conv8x1_line(x, W):  # includes ranks, files, and diags
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')


class NeuralNet:
    def __init__(self, load_weights=False, load_file=None):
        """
            Initialize neural net. Generates session, placeholders, variables, 
            and structure.
            Input:
                load_weights [Bool]:
                    If true, the neural net will load weights saved from a file
                    instead of initializing them from a normal distribution.
        """
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        # create session
        self.set_session()

        # create variables
        if load_weights:
            self.load_weight_values(load_file)
        else:
            self.initialize_tf_variables()

        self.sess.run(tf.initialize_all_variables())

        # create placeholders
        self.data = tf.placeholder(tf.float32, shape=[None, 8, 8, NUM_CHANNELS])
        self.data_diags = tf.placeholder(tf.float32, shape=[None, 10, 8, NUM_CHANNELS])
        self.true_value = tf.placeholder(tf.float32, shape=[None])

        # create neural net structure
        self.neural_net()

    def set_session(self):
        """ Sets tensorflow session """
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

    def initialize_tf_variables(self):
        """
            Initializes all weight variables to normal distribution, and all
            bias variables to a constant.
        """
        self.W_grid = weight_variable([5, 5, NUM_CHANNELS, NUM_FEAT])
        self.W_rank = weight_variable([8, 1, NUM_CHANNELS, NUM_FEAT])
        self.W_file = weight_variable([1, 8, NUM_CHANNELS, NUM_FEAT])
        self.W_diag = weight_variable([1, 8, NUM_CHANNELS, NUM_FEAT])

        # biases
        self.b_grid = bias_variable([NUM_FEAT])
        self.b_rank = bias_variable([NUM_FEAT])
        self.b_file = bias_variable([NUM_FEAT])
        self.b_diag = bias_variable([NUM_FEAT])

        # fully connected layer 1, weights + biases
        self.W_fc_1 = weight_variable([90 * NUM_FEAT, NUM_HIDDEN])
        self.b_fc_1 = bias_variable([NUM_HIDDEN])

        # fully connected layer 2, weights + biases
        self.W_fc_2 = weight_variable([NUM_HIDDEN, NUM_HIDDEN])
        self.b_fc_2 = bias_variable([NUM_HIDDEN])

        # Ouput layer
        self.W_final = weight_variable([NUM_HIDDEN, 1])
        self.b_final = bias_variable([1])

    def load_weight_values(self, filename='weights_1000.p'):
        """
            Sets all variables to values loaded from a file
            Input: 
                filename[String]:
                    Name of the file to load weight values from
        """
        pickle_path = self.dir_path + '/../pickles/' + filename
        weight_values = pickle.load(open(pickle_path, 'rb'))

        self.W_grid = tf.Variable(weight_values['W_grid'])
        self.W_rank = tf.Variable(weight_values['W_rank'])
        self.W_file = tf.Variable(weight_values['W_file'])
        self.W_diag = tf.Variable(weight_values['W_diag'])
        self.W_fc_1 = tf.Variable(weight_values['W_fc_1'])
        self.W_fc_2 = tf.Variable(weight_values['W_fc_2'])
        self.W_final = tf.Variable(weight_values['W_final'])

        self.b_grid = tf.Variable(weight_values['b_grid'])
        self.b_rank = tf.Variable(weight_values['b_rank'])
        self.b_file = tf.Variable(weight_values['b_file'])
        self.b_diag = tf.Variable(weight_values['b_diag'])
        self.b_fc_1 = tf.Variable(weight_values['b_fc_1'])
        self.b_fc_2 = tf.Variable(weight_values['b_fc_2'])
        self.b_final = tf.Variable(weight_values['b_final'])

    def save_weight_values(self, filename='weight_values.p'):
        ''' 
            Saves all variable values a pickle file
            Input: 
                filename[String]:
                    Name of the file to save weight values to
        '''
        weight_values = {}

        weight_values['W_grid'], weight_values['W_rank'], weight_values['W_file'], weight_values['W_diag'], \
        weight_values['W_fc_1'], weight_values['W_fc_2'], weight_values['W_final'], \
        weight_values['b_grid'], weight_values['b_rank'], weight_values['b_file'], weight_values['b_diag'], \
        weight_values['b_fc_1'], weight_values['b_fc_2'], weight_values['b_final'] = \
            self.sess.run([self.W_grid, self.W_rank, self.W_file, self.W_diag, self.W_fc_1, self.W_fc_2, self.W_final, \
                           self.W_grid, self.W_rank, self.W_file, self.W_diag, self.W_fc_1, self.W_fc_2, self.W_final])

        pickle_path = self.dir_path + '/../pickles/' + filename
        pickle.dump(weight_values, open(pickle_path, 'wb'))

    def neural_net(self):
        """
            Structure of neural net.
            Sets member variable 'pred_value' to the tensor representing the
            output of neural net.
        """
        batch_size = tf.shape(self.data)[0]

        # outputs to conv layer
        o_grid = tf.nn.relu(conv5x5_grid(self.data, self.W_grid) + self.b_grid)
        o_rank = tf.nn.relu(conv8x1_line(self.data, self.W_rank) + self.b_rank)
        o_file = tf.nn.relu(conv8x1_line(self.data, self.W_file) + self.b_file)
        o_diag = tf.nn.relu(conv8x1_line(self.data_diags, self.W_diag) + self.b_diag)

        o_grid = tf.reshape(o_grid, [batch_size, 64 * NUM_FEAT])
        o_rank = tf.reshape(o_rank, [batch_size, 8 * NUM_FEAT])
        o_file = tf.reshape(o_file, [batch_size, 8 * NUM_FEAT])
        o_diag = tf.reshape(o_diag, [batch_size, 10 * NUM_FEAT])

        o_conn = tf.concat(1, [o_grid, o_rank, o_file, o_diag])

        # output of fully connected layer 1
        o_fc_1 = tf.nn.relu(tf.matmul(o_conn, self.W_fc_1) + self.b_fc_1)

        # output of fully connected layer 2
        o_fc_2 = tf.nn.relu(tf.matmul(o_fc_1, self.W_fc_2) + self.b_fc_2)

        # final_output
        self.pred_value = tf.sigmoid(tf.matmul(o_fc_2, self.W_final) + self.b_final)

    # TODO S: Maybe combine the following two functions? I think this only gets used in guerilla.py but i'm not sure.
    def evaluate(self, fen):
        """
        Evaluates chess board.
             Input:
                 fen [String]:
                     FEN of chess board.
             Output:
                 Score between 0 (bad) and 1 (good).
        """

        # TODO S: Add board flipping if fen[1]='black' !
        fen = fen.split()[0]
        board = dc.fen_to_channels(fen)
        diagonal = dc.get_diagonals(board)

        board = np.array([board])
        diagonal = np.array([diagonal])
        return self.pred_value.eval(feed_dict={self.data: board, self.data_diags: diagonal}, session=self.sess)


    def evaluate_board(self, board):
        return self.evaluate(board.fen())
