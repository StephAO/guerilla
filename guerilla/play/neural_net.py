import pickle

import numpy as np
import tensorflow as tf
from pkg_resources import resource_filename

import guerilla.data_handler as dh
from guerilla.hyper_parameters import *


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
    training_modes = ['adagrad', 'adadelta', 'gradient_descent']

    def __init__(self, use_conv=True, load_file=None, training_mode=None, verbose=True):
        """
            Initializes neural net. Generates session, placeholders, variables,
            and structure.
            Input:
                load_weights [Bool]:
                    If true, the neural net will load weights saved from a file
                    instead of initializing them from a normal distribution.
                training_mode [String]
                    Training mode to be used. Defaults to Adagrad.
        """
        self.use_conv = use_conv

        self.load_file = load_file
        self.verbose = verbose

        if training_mode is None:
            training_mode = 'adagrad'
        elif training_mode not in NeuralNet.training_modes:
            raise ValueError("Invalid training mode input! Please refer to NeuralNet.training_modes for valid inputs.")
        self.training_mode = training_mode

        if self.verbose:
            print "Training neural net using %s." % self.training_mode

        # declare layer variables
        self.sess = None
        if self.use_conv:
            self.W_grid = None
            self.W_rank = None
            self.W_file = None
            self.W_diag = None
            self.b_grid = None
            self.b_rank = None
            self.b_file = None
            self.b_diag = None
        self.W_fc = [None] * hp['NUM_FC_LAYERS']
        self.b_fc = [None] * hp['NUM_FC_LAYERS']
        self.W_final = None
        self.b_final = None

        # declare output variable
        self.pred_value = None

        # tf session and variables
        self.sess = None

        self.conv_layer_size = 64 + 8 + 8 + 10  # 90

        # define all variables
        self.define_tf_variables()

        # all weights + biases
        # Currently the order is necessary for assignment operators
        self.all_weights = []

        if self.use_conv:
            self.all_weights.extend([self.W_grid, self.W_rank, self.W_file, self.W_diag])
        self.all_weights.extend(self.W_fc)
        self.all_weights.append(self.W_final)

        if self.use_conv:
            self.all_weights.extend([self.b_grid, self.b_rank, self.b_file, self.b_diag])
        self.all_weights.extend(self.b_fc)
        self.all_weights.append(self.b_final)

        if self.use_conv:
            # subsets of weights and biases
            self.base_weights = [self.W_grid, self.W_rank, self.W_file, self.W_diag]
            self.base_biases = [self.b_grid, self.b_rank, self.b_file, self.b_diag]

        # input placeholders
        self.data = tf.placeholder(tf.float32, shape=[None, 8, 8, hp['NUM_CHANNELS']])
        self.data_diags = tf.placeholder(tf.float32, shape=[None, 10, 8, hp['NUM_CHANNELS']])
        self.true_value = tf.placeholder(tf.float32, shape=[None])

        # assignment placeholders
        if self.use_conv:
            self.W_grid_placeholder = tf.placeholder(tf.float32, shape=[5, 5, hp['NUM_CHANNELS'], hp['NUM_FEAT']])
            self.W_rank_placeholder = tf.placeholder(tf.float32, shape=[1, 8, hp['NUM_CHANNELS'], hp['NUM_FEAT']])
            self.W_file_placeholder = tf.placeholder(tf.float32, shape=[8, 1, hp['NUM_CHANNELS'], hp['NUM_FEAT']])
            self.W_diag_placeholder = tf.placeholder(tf.float32, shape=[1, 8, hp['NUM_CHANNELS'], hp['NUM_FEAT']])

            self.b_grid_placeholder = tf.placeholder(tf.float32, shape=[hp['NUM_FEAT']])
            self.b_rank_placeholder = tf.placeholder(tf.float32, shape=[hp['NUM_FEAT']])
            self.b_file_placeholder = tf.placeholder(tf.float32, shape=[hp['NUM_FEAT']])
            self.b_diag_placeholder = tf.placeholder(tf.float32, shape=[hp['NUM_FEAT']])

        self.W_fc_placeholders = [None] * hp['NUM_FC_LAYERS']
        self.b_fc_placeholders = [None] * hp['NUM_FC_LAYERS']
        if self.use_conv:
            self.W_fc_placeholders[0] = tf.placeholder(tf.float32,
                                                       shape=[self.conv_layer_size * hp['NUM_FEAT'], hp['NUM_HIDDEN']])
        else:
            self.W_fc_placeholders[0] = tf.placeholder(tf.float32, shape=[8 * 8 * hp['NUM_CHANNELS'], hp['NUM_HIDDEN']])
        self.b_fc_placeholders[0] = tf.placeholder(tf.float32, shape=[hp['NUM_HIDDEN']])

        for i in xrange(1, hp['NUM_FC_LAYERS']):
            self.W_fc_placeholders[i] = tf.placeholder(tf.float32, shape=[hp['NUM_HIDDEN'], hp['NUM_HIDDEN']])
            self.b_fc_placeholders[i] = tf.placeholder(tf.float32, shape=[hp['NUM_HIDDEN']])

        self.W_final_placeholder = tf.placeholder(tf.float32, shape=[hp['NUM_HIDDEN'], 1])
        self.b_final_placeholder = tf.placeholder(tf.float32, shape=[1])

        # same order as all weights
        self.all_placeholders = []
        if self.use_conv:
            self.all_placeholders.extend(
                [self.W_grid_placeholder, self.W_rank_placeholder, self.W_file_placeholder, self.W_diag_placeholder])
        self.all_placeholders.extend(self.W_fc_placeholders)
        self.all_placeholders.append(self.W_final_placeholder)

        if self.use_conv:
            self.all_placeholders.extend(
                [self.b_grid_placeholder, self.b_rank_placeholder, self.b_file_placeholder, self.b_diag_placeholder])
        self.all_placeholders.extend(self.b_fc_placeholders)
        self.all_placeholders.append(self.b_final_placeholder)

        # create assignment operators
        if self.use_conv:
            self.W_grid_assignment = self.W_grid.assign(self.W_grid_placeholder)
            self.W_rank_assignment = self.W_rank.assign(self.W_rank_placeholder)
            self.W_file_assignment = self.W_file.assign(self.W_file_placeholder)
            self.W_diag_assignment = self.W_diag.assign(self.W_diag_placeholder)

            self.b_grid_assignment = self.b_grid.assign(self.b_grid_placeholder)
            self.b_rank_assignment = self.b_rank.assign(self.b_rank_placeholder)
            self.b_file_assignment = self.b_file.assign(self.b_file_placeholder)
            self.b_diag_assignment = self.b_diag.assign(self.b_diag_placeholder)

        self.W_fc_assignments = [None] * hp['NUM_FC_LAYERS']
        self.b_fc_assignments = [None] * hp['NUM_FC_LAYERS']
        for i in xrange(hp['NUM_FC_LAYERS']):
            self.W_fc_assignments[i] = (self.W_fc[i].assign(self.W_fc_placeholders[i]))
            self.b_fc_assignments[i] = (self.b_fc[i].assign(self.b_fc_placeholders[i]))

        self.W_final_assignment = self.W_final.assign(self.W_final_placeholder)
        self.b_final_assignment = self.b_final.assign(self.b_final_placeholder)

        # same order as all weights and all placeholders
        self.all_assignments = []

        if self.use_conv:
            self.all_assignments.extend(
                [self.W_grid_assignment, self.W_rank_assignment, self.W_file_assignment, self.W_diag_assignment])
        self.all_assignments.extend(self.W_fc_assignments)
        self.all_assignments.append(self.W_final_assignment)

        if self.use_conv:
            self.all_assignments.extend(
                [self.b_grid_assignment, self.b_rank_assignment, self.b_file_assignment, self.b_diag_assignment])
        self.all_assignments.extend(self.b_fc_assignments)
        self.all_assignments.append(self.b_final_assignment)

        # create neural net graph
        self.neural_net()

        # gradient op and placeholder (must be defined after self.pred_value is defined)
        self.grad_all_op = tf.gradients(self.pred_value, self.all_weights)

        # Define training operators and variables
        # Using MAE since value difference will always be 0 <= x <= 1, don't want the sublinear error when using MSE
        #   Note: Ensures that both inputs are the same shape
        self.MAE = tf.reduce_sum(tf.abs(tf.sub(
            tf.reshape(self.pred_value, shape=tf.shape(self.true_value)), self.true_value)))

        self.MSE = tf.reduce_sum(tf.pow(tf.sub(
            tf.reshape(self.pred_value, shape=tf.shape(self.true_value)), self.true_value), 2))

        if self.training_mode == 'adagrad':
            self.train_optimizer = tf.train.AdagradOptimizer(hp['LEARNING_RATE'])
        elif self.training_mode == 'adadelta':
            self.train_optimizer = tf.train.AdadeltaOptimizer(learning_rate=hp['LEARNING_RATE'], rho=hp['DECAY_RATE'])
        elif self.training_mode == 'gradient_descent':
            self.train_optimizer = tf.train.GradientDescentOptimizer(hp['LEARNING_RATE'])
        self.train_step = self.train_optimizer.minimize(self.MSE)
        self.train_saver = tf.train.Saver(
            var_list=self.get_training_vars())  # TODO: Combine var saving with "in_training" weight saving

    def init_graph(self):
        """
        Initializes the weights and assignment ops of the neural net, either from a file or a truncated Gaussian.
        Note: The session must be open beforehand.
        """
        assert self.sess is not None

        # initialize or load variables
        if self.load_file:
            self.load_weight_values(self.load_file)
            # Initialize un-initialized variables (non-weight variables)
            self.sess.run(tf.initialize_variables(set(tf.all_variables()) - set(self.all_weights)))
        else:
            if self.verbose:
                print "Initializing variables from a normal distribution."
            self.sess.run(tf.initialize_all_variables())

    def start_session(self):
        """ Starts tensorflow session """

        assert self.sess is None

        self.sess = tf.Session()
        if self.verbose:
            print "Tensorflow session opened."

    def close_session(self):
        """ Closes tensorflow session"""
        assert self.sess is not None  # M: Not sure if this should be an assert

        self.sess.close()
        if self.verbose:
            print "Tensorflow session closed."

    def define_tf_variables(self):
        """
            Initializes all weight variables to normal distribution, and all
            bias variables to a constant.
        """
        if self.use_conv:
            # conv weights
            self.W_grid = weight_variable([5, 5, hp['NUM_CHANNELS'], hp['NUM_FEAT']])
            self.W_rank = weight_variable([1, 8, hp['NUM_CHANNELS'], hp['NUM_FEAT']])
            self.W_file = weight_variable([8, 1, hp['NUM_CHANNELS'], hp['NUM_FEAT']])
            self.W_diag = weight_variable([1, 8, hp['NUM_CHANNELS'], hp['NUM_FEAT']])

            # conv biases
            self.b_grid = bias_variable([hp['NUM_FEAT']])
            self.b_rank = bias_variable([hp['NUM_FEAT']])
            self.b_file = bias_variable([hp['NUM_FEAT']])
            self.b_diag = bias_variable([hp['NUM_FEAT']])

        # fully connected layer 1, weights + biases
        if self.use_conv:
            self.W_fc[0] = weight_variable([self.conv_layer_size * hp['NUM_FEAT'], hp['NUM_HIDDEN']])
        else:
            self.W_fc[0] = weight_variable([8 * 8 * hp['NUM_CHANNELS'], hp['NUM_HIDDEN']])
        self.b_fc[0] = bias_variable([hp['NUM_HIDDEN']])

        for i in xrange(1, hp['NUM_FC_LAYERS']):
            # fully connected layer n, weights + biases
            self.W_fc[i] = weight_variable([hp['NUM_HIDDEN'], hp['NUM_HIDDEN']])
            self.b_fc[i] = bias_variable([hp['NUM_HIDDEN']])

        # Output layer
        self.W_final = weight_variable([hp['NUM_HIDDEN'], 1])
        self.b_final = bias_variable([1])

    def get_training_vars(self):
        """
        Returns the training variables associated with the current training mode.
        Returns None if there are no associated variables.
        Output:
            var_dict [Dict] or [None]:
                Dictionary of variables.
        """
        if self.training_mode == 'gradient_descent':
            return None
        else:
            var_dict = dict()
            train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            slot_names = self.train_optimizer.get_slot_names()  # Result should just be accumulator
            for name in slot_names:
                for var in train_vars:
                    val = self.train_optimizer.get_slot(var, name)
                    if val:
                        var_dict[var.name] = val
            return var_dict

    def save_training_vars(self, path):
        """
        Saves the training variables associated with the current training mode to a file in path.
        Returns the file name.
        Input:
            path [String]
                Path specifying where the variables should be saved.
        Ouput:
            filename [String]
                Filename specifying where the training variables were saved.
        """
        filename = None
        if self.training_mode == 'gradient_descent':
            pass
        else:
            filename = self.train_saver.save(self.sess, path)

        if self.verbose:
            print "Saved training vars to %s" % filename
        return filename

    def load_training_vars(self, filename):
        """
        Loads the training variable associated with the current training mode.
        Input:
            filename [String]
                Filename where training variables are stored.
        """
        if self.training_mode == 'gradient_descent':
            pass
        else:
            self.train_saver.restore(self.sess, filename)

        if self.verbose:
            print "Loaded training vars from %s " % filename

    def load_weight_values(self, _filename='weight_values.p'):
        """
            Sets all variables to values loaded from a file
            Input: 
                filename[String]:
                    Name of the file to load weight values from
        """

        pickle_path = resource_filename('guerilla', 'data/weights/' + _filename)
        if self.verbose:
            print "Loading weights values from %s" % pickle_path
        values_dict = pickle.load(open(pickle_path, 'rb'))

        weight_values = []
        if self.use_conv:
            weight_values.extend([values_dict['W_grid'], values_dict['W_rank'],
                                  values_dict['W_file'], values_dict['W_diag']])
        weight_values.extend(values_dict['W_fc'])
        weight_values.append(values_dict['W_final'])

        if self.use_conv:
            weight_values.extend([values_dict['b_grid'], values_dict['b_rank'],
                                  values_dict['b_file'], values_dict['b_diag']])
        weight_values.extend(values_dict['b_fc'])
        weight_values.append(values_dict['b_final'])

        self.set_all_weights(weight_values)

    def save_weight_values(self, _filename='weight_values.p'):
        """
            Saves all variable values a pickle file
            Input: 
                filename[String]:
                    Name of the file to save weight values to
        """

        pickle_path = resource_filename('guerilla', 'data/weights/' + _filename)
        pickle.dump(self.get_weight_values(), open(pickle_path, 'wb'))
        if self.verbose:
            print "Weights saved to %s" % _filename

    def get_weight_values(self):
        """ 
            Returns values of weights as a dictionary
        """
        weight_values = dict()

        if self.use_conv:
            weight_values['W_grid'], weight_values['W_rank'], weight_values['W_file'], \
            weight_values['W_diag'], \
            weight_values['b_grid'], weight_values['b_rank'], weight_values['b_file'], \
            weight_values['b_diag'] = \
                self.sess.run([self.W_grid, self.W_rank, self.W_file, self.W_diag,
                               self.b_grid, self.b_rank, self.b_file, self.b_diag])

        # Set up dict entries for fully-connected layers
        weight_values['W_fc'] = [None] * len(self.W_fc)
        weight_values['b_fc'] = [None] * len(self.b_fc)

        # Get weight values
        result = self.sess.run(self.W_fc + self.b_fc + [self.W_final, self.b_final])

        # Assign fully connected layers
        for i in range(len(self.W_fc)):
            weight_values['W_fc'][i] = result[i]
            weight_values['b_fc'][i] = result[len(self.W_fc) + i]

        # Assign final layer
        weight_values['W_final'] = result[-2]
        weight_values['b_final'] = result[-1]

        return weight_values

    def neural_net(self):
        """
            Structure of neural net.
            Sets member variable 'pred_value' to the tensor representing the
            output of neural net.
        """
        batch_size = tf.shape(self.data)[0]

        o_fc = [None] * hp['NUM_FC_LAYERS']

        if self.use_conv:
            o_grid = tf.nn.relu(conv5x5_grid(self.data, self.W_grid) + self.b_grid)
            o_rank = tf.nn.relu(conv8x1_line(self.data, self.W_rank) + self.b_rank)
            o_file = tf.nn.relu(conv8x1_line(self.data, self.W_file) + self.b_file)
            o_diag = tf.nn.relu(conv8x1_line(self.data_diags, self.W_diag) + self.b_diag)

            o_grid = tf.reshape(o_grid, [batch_size, 64 * hp['NUM_FEAT']])
            o_rank = tf.reshape(o_rank, [batch_size, 8 * hp['NUM_FEAT']])
            o_file = tf.reshape(o_file, [batch_size, 8 * hp['NUM_FEAT']])
            o_diag = tf.reshape(o_diag, [batch_size, 10 * hp['NUM_FEAT']])

            # output of convolutional layer
            o_conn = tf.concat(1, [o_grid, o_rank, o_file, o_diag])

            # output of fully connected layer 1
            o_fc[0] = tf.nn.relu(tf.matmul(o_conn, self.W_fc[0]) + self.b_fc[0])

        else:
            data = tf.reshape(self.data, [batch_size, 64 * hp['NUM_CHANNELS']])
            # output of fully connected layer 1
            o_fc[0] = tf.nn.relu(tf.matmul(data, self.W_fc[0]) + self.b_fc[0])

        for i in xrange(1, hp['NUM_FC_LAYERS']):
            # output of fully connected layer n
            o_fc[i] = tf.nn.relu(tf.matmul(o_fc[i - 1], self.W_fc[i]) + self.b_fc[i])

        # final_output
        self.pred_value = tf.sigmoid(tf.matmul(o_fc[-1], self.W_final) + self.b_final)

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
                    List of values with which to update weights. Must be in desired order.
        """
        assert len(weight_vals) == len(self.all_weights)

        # match value to placeholders
        placeholder_dict = dict()
        for i, placeholder in enumerate(self.all_placeholders):
            placeholder_dict[placeholder] = weight_vals[i]

        # Run assignment/update
        self.sess.run(self.all_assignments, feed_dict=placeholder_dict)

    def add_all_weights(self, weight_vals):  # TODO rename to add_to_all_weights
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

    # TODO S: Maybe combine the following two functions? I think this only gets used in Guerilla but i'm not sure.
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
