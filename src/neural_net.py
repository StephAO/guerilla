import tensorflow as tf
import numpy as np
import pickle
import os
from hyper_parameters import *
import data_handler as dh


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv5x5_grid(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')  # Pad or fit? (same is pad, fit is valid)


def conv8x1_line(x, w):  # includes ranks, files, and diagonals
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='VALID')


class NeuralNet:
    def __init__(self, load_file=None):
        """
            Initializes neural net. Generates session, placeholders, variables,
            and structure.
            Input:
                load_weights [Bool]:
                    If true, the neural net will load weights saved from a file
                    instead of initializing them from a normal distribution.
        """
        self.dir_path = os.path.dirname(__file__)

        self.load_file = load_file

        # declare layer variables
        self.sess = None
        self.W_grid = None
        self.W_rank = None
        self.W_file = None
        self.W_diag = None
        self.b_grid = None
        self.b_rank = None
        self.b_file = None
        self.b_diag = None
        self.W_fc_1 = None
        self.b_fc_1 = None
        self.W_fc_2 = None
        self.b_fc_2 = None
        self.W_final = None
        self.b_final = None

        # declare output variable
        self.pred_value = None

        # tf session and variables
        self.sess = None

        # define all variables
        self.define_tf_variables()

        # all weights + biases
        # Currently the order is necessary for assignment operators
        self.all_weights = [self.W_grid, self.W_rank, self.W_file, self.W_diag, self.W_fc_1, self.W_fc_2, self.W_final,
                            self.b_grid, self.b_rank, self.b_file, self.b_diag, self.b_fc_1, self.b_fc_2, self.b_final]

        # subsets of weights and biases
        self.base_weights = [self.W_grid, self.W_rank, self.W_file, self.W_diag]
        self.base_biases = [self.b_grid, self.b_rank, self.b_file, self.b_diag]

        # input placeholders
        self.data = tf.placeholder(tf.float32, shape=[None, 8, 8, NUM_CHANNELS])
        self.data_diags = tf.placeholder(tf.float32, shape=[None, 10, 8, NUM_CHANNELS])
        self.true_value = tf.placeholder(tf.float32, shape=[None])

        # assignment placeholders
        self.W_grid_placeholder = tf.placeholder(tf.float32, shape=[5, 5, NUM_CHANNELS, NUM_FEAT])
        self.W_rank_placeholder = tf.placeholder(tf.float32, shape=[1, 8, NUM_CHANNELS, NUM_FEAT])
        self.W_file_placeholder = tf.placeholder(tf.float32, shape=[8, 1, NUM_CHANNELS, NUM_FEAT])
        self.W_diag_placeholder = tf.placeholder(tf.float32, shape=[1, 8, NUM_CHANNELS, NUM_FEAT])

        self.b_grid_placeholder = tf.placeholder(tf.float32, shape=[NUM_FEAT])
        self.b_rank_placeholder = tf.placeholder(tf.float32, shape=[NUM_FEAT])
        self.b_file_placeholder = tf.placeholder(tf.float32, shape=[NUM_FEAT])
        self.b_diag_placeholder = tf.placeholder(tf.float32, shape=[NUM_FEAT])

        self.W_fc1_placeholder = tf.placeholder(tf.float32, shape=[90 * NUM_FEAT, NUM_HIDDEN])
        self.b_fc1_placeholder = tf.placeholder(tf.float32, shape=[NUM_HIDDEN])
        self.W_fc2_placeholder = tf.placeholder(tf.float32, shape=[NUM_HIDDEN, NUM_HIDDEN])
        self.b_fc2_placeholder = tf.placeholder(tf.float32, shape=[NUM_HIDDEN])

        self.W_final_placeholder = tf.placeholder(tf.float32, shape=[NUM_HIDDEN, 1])
        self.b_final_placeholder = tf.placeholder(tf.float32, shape=[1])

        # same order as all weights
        self.all_placeholders = \
            [self.W_grid_placeholder, self.W_rank_placeholder, self.W_file_placeholder, self.W_diag_placeholder,
             self.W_fc1_placeholder, self.W_fc2_placeholder, self.W_final_placeholder,
             self.b_grid_placeholder, self.b_rank_placeholder, self.b_file_placeholder, self.b_diag_placeholder,
             self.b_fc1_placeholder, self.b_fc2_placeholder, self.b_final_placeholder]

        # create assignment operators
        self.W_grid_assignment = self.W_grid.assign(self.W_grid_placeholder)
        self.W_rank_assignment = self.W_rank.assign(self.W_rank_placeholder)
        self.W_file_assignment = self.W_file.assign(self.W_file_placeholder)
        self.W_diag_assignment = self.W_diag.assign(self.W_diag_placeholder)

        self.b_grid_assignment = self.b_grid.assign(self.b_grid_placeholder)
        self.b_rank_assignment = self.b_rank.assign(self.b_rank_placeholder)
        self.b_file_assignment = self.b_file.assign(self.b_file_placeholder)
        self.b_diag_assignment = self.b_diag.assign(self.b_diag_placeholder)

        self.W_fc1_assignment = self.W_fc_1.assign(self.W_fc1_placeholder)
        self.b_fc1_assignment = self.b_fc_1.assign(self.b_fc1_placeholder)
        self.W_fc2_assignment = self.W_fc_2.assign(self.W_fc2_placeholder)
        self.b_fc2_assignment = self.b_fc_2.assign(self.b_fc2_placeholder)

        self.W_final_assignment = self.W_final.assign(self.W_final_placeholder)
        self.b_final_assignment = self.b_final.assign(self.b_final_placeholder)

        # same order as all weights and all placeholders
        self.all_assignments = \
            [self.W_grid_assignment, self.W_rank_assignment, self.W_file_assignment, self.W_diag_assignment,
             self.W_fc1_assignment, self.W_fc2_assignment, self.W_final_assignment,
             self.b_grid_assignment, self.b_rank_assignment, self.b_file_assignment, self.b_diag_assignment,
             self.b_fc1_assignment, self.b_fc2_assignment, self.b_final_assignment]

        # create neural net graph
        self.neural_net()

        # gradient op and placeholder (must be defined after self.pred_value is defined)
        self.grad_all_op = tf.gradients(self.pred_value, self.all_weights)

    def init_graph(self):
        """
        Initializes the weights and assignment ops of the neural net, either from a file or a truncated Gaussian.
        Note: The session must be open beforehand.
        """
        assert self.sess is not None

        # initialize or load variables
        if self.load_file:
            self.load_weight_values(self.load_file)
        else:
            print "Initializing variables from a normal distribution."
            self.sess.run(tf.initialize_all_variables())

    def start_session(self):
        """ Starts tensorflow session """

        assert self.sess is None

        self.sess = tf.Session()
        print "Tensorflow session opened."

    def close_session(self):
        """ Closes tensorflow session"""
        assert self.sess is not None # M: Not sure if this should be an assert

        self.sess.close()
        print "Tensorflow session closed."

    def define_tf_variables(self):
        """
            Initializes all weight variables to normal distribution, and all
            bias variables to a constant.
        """

        self.W_grid = weight_variable([5, 5, NUM_CHANNELS, NUM_FEAT])
        self.W_rank = weight_variable([1, 8, NUM_CHANNELS, NUM_FEAT])
        self.W_file = weight_variable([8, 1, NUM_CHANNELS, NUM_FEAT])
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

        # Output layer
        self.W_final = weight_variable([NUM_HIDDEN, 1])
        self.b_final = bias_variable([1])

    def load_weight_values(self, _filename='weight_values.p'):
        """
            Sets all variables to values loaded from a file
            Input: 
                filename[String]:
                    Name of the file to load weight values from
        """

        pickle_path = self.dir_path + '/../pickles/' + _filename
        print "Loading weights values from %s" % pickle_path
        weight_values = pickle.load(open(pickle_path, 'rb'))

        weight_values = [weight_values['W_grid'], weight_values['W_rank'],
                         weight_values['W_file'], weight_values['W_diag'],
                         weight_values['W_fc_1'], weight_values['W_fc_2'],
                         weight_values['W_final'],
                         weight_values['b_grid'], weight_values['b_rank'],
                         weight_values['b_file'], weight_values['b_diag'],
                         weight_values['b_fc_1'], weight_values['b_fc_2'],
                         weight_values['b_final']]

        self.set_all_weights(weight_values)

    def save_weight_values(self, _filename='weight_values.p'):
        """
            Saves all variable values a pickle file
            Input: 
                filename[String]:
                    Name of the file to save weight values to
        """

        pickle_path = self.dir_path + '/../pickles/' + _filename
        pickle.dump(self.get_weight_values(), open(pickle_path, 'wb'))
        print "Weights saved to %s" % _filename

    def get_weight_values(self):
        """ 
            Returns values of weights as a dictionary
        """
        weight_values = dict()

        weight_values['W_grid'], weight_values['W_rank'], weight_values['W_file'], weight_values['W_diag'], \
            weight_values['W_fc_1'], weight_values['W_fc_2'], weight_values['W_final'], \
            weight_values['b_grid'], weight_values['b_rank'], weight_values['b_file'], weight_values['b_diag'], \
            weight_values['b_fc_1'], weight_values['b_fc_2'], weight_values['b_final'] = \
            self.sess.run([self.W_grid, self.W_rank, self.W_file, self.W_diag, self.W_fc_1, self.W_fc_2, self.W_final,
                           self.b_grid, self.b_rank, self.b_file, self.b_diag, self.b_fc_1, self.b_fc_2, self.b_final])

        return weight_values

    def neural_net(self):
        """
            Structure of neural net.
            Sets member variable 'pred_value' to the tensor representing the
            output of neural net.
        """
        batch_size = tf.shape(self.data)[0]

        # outputs to convolutional layer
        a = conv5x5_grid(self.data, self.W_grid)
        # print tf.shape(self.b_grid).eval()
        b = tf.nn.relu(a + self.b_grid)
        o_grid = tf.nn.relu(conv5x5_grid(self.data, self.W_grid) + self.b_grid)
        o_rank = tf.nn.relu(conv8x1_line(self.data, self.W_rank) + self.b_rank)
        o_file = tf.nn.relu(conv8x1_line(self.data, self.W_file) + self.b_file)
        o_diag = tf.nn.relu(conv8x1_line(self.data_diags, self.W_diag) + self.b_diag)

        o_grid = tf.reshape(o_grid, [batch_size, 64 * NUM_FEAT])
        o_rank = tf.reshape(o_rank, [batch_size, 8 * NUM_FEAT])
        o_file = tf.reshape(o_file, [batch_size, 8 * NUM_FEAT])
        o_diag = tf.reshape(o_diag, [batch_size, 10 * NUM_FEAT])

        # output of convolutional layer
        o_conn = tf.concat(1, [o_grid, o_rank, o_file, o_diag])

        # output of fully connected layer 1
        o_fc_1 = tf.nn.relu(tf.matmul(o_conn, self.W_fc_1) + self.b_fc_1)

        # output of fully connected layer 2
        o_fc_2 = tf.nn.relu(tf.matmul(o_fc_1, self.W_fc_2) + self.b_fc_2)

        # final_output
        self.pred_value = tf.sigmoid(tf.matmul(o_fc_2, self.W_final) + self.b_final)

    def get_weights(self, weight_vars):
        """
        Get the weight values of the input.
            Input:
                weight_vars [List]
                    List of weights to get.
            Output:
                weights [List]
                    Weights & biases.
        """
        return self.sess.run(weight_vars)

    def set_all_weights(self, weight_vals):
        """
        NOTE: currently only supports updating all weights, must be in the same order.
        Updates the neural net weights based on the input.
            Input:
                weight_vals [List]
                    List of values with which to update weights. Must be iin desired order.
        """
        assert len(weight_vals) == len(self.all_weights)

        # match value to placeholders
        placeholder_dict = dict()
        for i, placeholder in enumerate(self.all_placeholders):
            placeholder_dict[placeholder] = weight_vals[i]

        # Run assignment/update
        self.sess.run(self.all_assignments, feed_dict=placeholder_dict)

    def add_all_weights(self, weight_vals):
        """
        Increments all the weight values by the input amount.
            Input:
                weight_vals [List]
                    List of values with which to update weights. Must be in desired order.
        """

        old_weights = self.get_weights(self.all_weights)
        new_weights = [old_weights[i] + weight_vals[i] for i in range(len(weight_vals))]

        self.set_all_weights(new_weights)

    def get_all_weights_gradient(self, fen):
        """
        Returns the gradient of the neural net at the output node, with respect to all weights.
        board.
            Input:
                fen [String]
                    FEN of board where gradient is to be taken. Must be for white playing next.
            Output:
                Gradient [List of floats].
        """

        if dh.black_is_next(fen):
            raise ValueError("Invalid gradient input, white must be next to play.")

        # calculate gradient
        return self.sess.run(self.grad_all_op, feed_dict=self.board_to_feed(fen))

    def board_to_feed(self, fen):
        """
        Converts the FEN of a SINGLE board to the required feed input for the neural net.
            Input:
                board [String]
                    FEN of board.
            Output:
                feed_dict [Dictionary]
                    Formatted input for neural net.
        """

        fen = fen.split()[0]
        board = dh.fen_to_channels(fen)
        diagonal = dh.get_diagonals(board)

        board = np.array([board])
        diagonal = np.array([diagonal])

        return {self.data: board, self.data_diags: diagonal}

    # TODO S: Maybe combine the following two functions? I think this only gets used in guerilla.py but i'm not sure.
    def evaluate(self, fen):
        """
        Evaluates chess board.
             Input:
                 fen [String]:
                     FEN of chess board.
             Output:
                 Score between 0 and 1. Represents probability of White (current player) winning.
        """
        if dh.black_is_next(fen):
            raise ValueError("Invalid evaluate input, white must be next to play.")

        return self.pred_value.eval(feed_dict=self.board_to_feed(fen), session=self.sess)[0][0]

    def evaluate_board(self, board):
        return self.evaluate(board.fen())


if __name__ == 'main':
    pass