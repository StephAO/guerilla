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
    # TODO: Add close session function to release resources.
    def __init__(self, load_weights=False, load_file=None):
        """
            Initializes neural net. Generates session, placeholders, variables,
            and structure.
            Input:
                load_weights [Bool]:
                    If true, the neural net will load weights saved from a file
                    instead of initializing them from a normal distribution.
        """
        self.dir_path = os.path.dirname(__file__)

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

        # declare other variables
        self.pred_value = None

        # create session
        self.set_session()

        # initialize variables
        if load_weights:
            assert load_file is not None, "Could not load weights, file has not been specified."
            self.load_weight_values(load_file)
        else:
            self.initialize_tf_variables()

        self.sess.run(tf.initialize_all_variables())

        # all weights + biases
        self.all_weights = [self.W_grid, self.W_rank, self.W_file, self.W_diag, self.W_fc_1, self.W_fc_2, self.W_final,
                            self.b_grid, self.b_rank, self.b_file, self.b_diag, self.b_fc_1, self.b_fc_2, self.b_final]

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
            self.sess = tf.InteractiveSession()  # TODO S: Why not just tf.Session()?

    def initialize_tf_variables(self):
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

    def load_weight_values(self, filename='weight_values.p'):
        """
            Sets all variables to values loaded from a file
            Input: 
                filename[String]:
                    Name of the file to load weight values from
        """
        print "Loading weight values..."
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
        """
            Saves all variable values a pickle file
            Input: 
                filename[String]:
                    Name of the file to save weight values to
        """

        weight_values = dict()

        weight_values['W_grid'], weight_values['W_rank'], weight_values['W_file'], weight_values['W_diag'], \
            weight_values['W_fc_1'], weight_values['W_fc_2'], weight_values['W_final'], \
            weight_values['b_grid'], weight_values['b_rank'], weight_values['b_file'], weight_values['b_diag'], \
            weight_values['b_fc_1'], weight_values['b_fc_2'], weight_values['b_final'] = \
            self.sess.run([self.W_grid, self.W_rank, self.W_file, self.W_diag, self.W_fc_1, self.W_fc_2, self.W_final,
                           self.b_grid, self.b_rank, self.b_file, self.b_diag, self.b_fc_1, self.b_fc_2, self.b_final])

        pickle_path = self.dir_path + '/../pickles/' + filename
        pickle.dump(weight_values, open(pickle_path, 'wb'))

    def neural_net(self):
        """
            Structure of neural net.
            Sets member variable 'pred_value' to the tensor representing the
            output of neural net.
        """
        batch_size = tf.shape(self.data)[0]

        # outputs to convolutional layer
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

    def update_weights(self, weight_vars, weight_vals):
        """
        Updates the neural net weights based on the input.
            Input:
                weight_vars [List]
                    List of weights to be updated
                weight_vals [List]
                    List of values with which to update weights. Must be in same order!
        """
        assert len(weight_vars) == len(weight_vals)

        # Create assignment for each weight
        num_weights = len(weight_vals)
        assignments = [None] * num_weights
        for i in range(num_weights):
            assignments[i] = weight_vars[i].assign(weight_vals[i])

        # Run assignment/update
        print ([str(x.eval()) for x in weight_vars])
        self.sess.run(assignments)
        print ([str(x.eval()) for x in weight_vars])


    def get_gradient(self, fen, weights):
        """
        Returns the gradient of the neural net at the output node, with respect to the specified weights.
        board.
            Input:
                fen [String]
                    FEN of board where gradient is to be taken.
                weights [List]
                    List of weight variables.
            Output:
                Gradient [List of floats].
        """

        #  declare gradient of predicted (output) value w.r.t. weights + biases
        grad = tf.gradients(self.pred_value, weights)

        # calculate gradient
        return self.sess.run(grad, feed_dict=self.board_to_feed(fen))

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

        return {self.data: board, self. data_diags: diagonal}

    # TODO S: Maybe combine the following two functions? I think this only gets used in guerilla.py but i'm not sure.
    def evaluate(self, fen):
        """
        Evaluates chess board.
             Input:
                 fen [String]:
                     FEN of chess board.
             Output:
                 Score between 0 (bad) and 1 (good). Represents probability of White (current player) winning.
        """

        if dh.fen_is_black(fen): raise ValueError("Invalid evaluate input, white must be next to play.")

        return self.pred_value.eval(feed_dict=self.board_to_feed(fen), session=self.sess)[0][0]

    def evaluate_board(self, board):
        return self.evaluate(board.fen())