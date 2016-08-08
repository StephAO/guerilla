import tensorflow as tf
import numpy as np
import pickle
from hyper_parameters import *


def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
        return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def conv5x5_grid(x, W):
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME') # Pad or fit? (same is pad, fit is valid)

def conv8x1_line(x, W): # includes ranks, files, and diags
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='VALID')


class NeuralNet:

    def __init__(self, load_weights=False):
        ''' 
            Initialize neural net. Generates session, placeholders, variables, 
            and structure.
            Input:
                load_weights [Bool]:
                    If true, the neural net will load weights saved from a file
                    instead of initializing them from a normal distribution.
        '''
        # create session
        self.set_session()

        # create variables
        if load_weights:
            self.load_weight_values()
        else:
            self.initialize_tf_variables()

        self.sess.run(tf.initialize_all_variables())

        # create placeholders
        self.data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,8,8,NUM_CHANNELS])
        self.data_diags = tf.placeholder(tf.float32, shape=[BATCH_SIZE,10,8,NUM_CHANNELS])
        self.true_value = tf.placeholder(tf.float32, shape=[BATCH_SIZE])

        # create neural net structure
        self.neural_net()

    def set_session(self):
        ''' Sets tensorflow session '''
        self.sess = tf.get_default_session()
        if self.sess is None:
            self.sess = tf.InteractiveSession()

    def initialize_tf_variables(self):
        ''' 
            Initializes all weight variables to normal distribution, and all
            bias variables to a constant.
        '''
        self.W_grid = weight_variable([5,5,NUM_CHANNELS,NUM_FEAT])
        self.W_rank = weight_variable([8,1,NUM_CHANNELS,NUM_FEAT])
        self.W_file = weight_variable([1,8,NUM_CHANNELS,NUM_FEAT])
        self.W_diag = weight_variable([1,8,NUM_CHANNELS,NUM_FEAT])

        # biases
        self.b_grid = bias_variable([NUM_FEAT])
        self.b_rank = bias_variable([NUM_FEAT])
        self.b_file = bias_variable([NUM_FEAT])
        self.b_diag = bias_variable([NUM_FEAT])

        # fully connected layer 1, weights + biases
        self.W_fc_1 = weight_variable([90*NUM_FEAT, NUM_HIDDEN])
        self.b_fc_1 = bias_variable([NUM_HIDDEN])

        # fully connected layer 2, weights + biases
        self.W_fc_2 = weight_variable([NUM_HIDDEN, NUM_HIDDEN])
        self.b_fc_2 = bias_variable([NUM_HIDDEN])

        # Ouput layer
        self.W_final = weight_variable([NUM_HIDDEN,1])
        self.b_final = bias_variable([1])
        
    def load_weight_values(self, filename = 'weight_values.p'):
        ''' 
            Sets all variables to values loaded from a file
            Input: 
                filename[String]:
                    Name of the file to load weight values from
        '''
        weight_values = pickle.load(open(filename, 'rb'))

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

    def save_weight_values(self, filename = 'weight_values.p'):
        ''' 
            Saves all variable values a pickle file
            Input: 
                filename[String]:
                    Name of the file to save weight values to
        '''
        weight_values = {}

        weight_values['W_grid'], weight_values['W_rank'], weight_values['W_file'], weight_values['W_diag'], weight_values['W_fc_1'], weight_values['W_fc_2'], weight_values['W_final'], \
        weight_values['b_grid'], weight_values['b_rank'], weight_values['b_file'], weight_values['b_diag'], weight_values['b_fc_1'], weight_values['b_fc_2'], weight_values['b_final'] = \
        self.sess.run( [ self.W_grid, self.W_rank, self.W_file, self.W_diag, self.W_fc_1, self.W_fc_2, self.W_final, \
                    self.W_grid, self.W_rank, self.W_file, self.W_diag, self.W_fc_1, self.W_fc_2, self.W_final ] )

        pickle.dump(weight_values, open(filename, 'wb'))

    def neural_net(self):
        '''
            Structure of neural net.
            Sets member variable 'pred_value' to the tensor representing the
            output of neural net.
        '''
        # outputs to conv layer
        o_grid = tf.nn.relu(conv5x5_grid(self.data, self.W_grid) + self.b_grid)
        o_rank = tf.nn.relu(conv8x1_line(self.data, self.W_rank) + self.b_rank)
        o_file = tf.nn.relu(conv8x1_line(self.data, self.W_file) + self.b_file)
        o_diag = tf.nn.relu(conv8x1_line(self.data_diags, self.W_diag) + self.b_diag)

        o_grid = tf.reshape(o_grid, [BATCH_SIZE, 64*NUM_FEAT])
        o_rank = tf.reshape(o_rank, [BATCH_SIZE, 8*NUM_FEAT])
        o_file = tf.reshape(o_file, [BATCH_SIZE, 8*NUM_FEAT])
        o_diag = tf.reshape(o_diag, [BATCH_SIZE, 10*NUM_FEAT])

        o_conn = tf.concat(1, [o_grid, o_rank, o_file, o_diag])

        # output of fully connected layer 1
        o_fc_1 = tf.nn.relu(tf.matmul(o_conn, self.W_fc_1) + self.b_fc_1)

        # output of fully connected layer 2
        o_fc_2 = tf.nn.relu(tf.matmul(o_fc_1, self.W_fc_2) + self.b_fc_2)

        # final_output
        self.pred_value = tf.sigmoid(tf.matmul(o_fc_2, self.W_final) + self.b_final)

def train(nn, boards, diagonals, true_values, save_weights=True):
    '''
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
    '''

    # From my limited understanding x_entropy is not suitable - but if im wrong it could be better
    # Using squared error instead
    cost = tf.reduce_sum(tf.pow(tf.sub(nn.pred_value, nn.true_value), 2))

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

    for i in xrange(np.shape(boards)[0]):  
        nn.sess.run([train_step], feed_dict={nn.data: boards[i], nn.data_diags: diagonals[i], nn.true_value: true_values[i]})

    if save_weights:
        nn.save_weight_values()

def evaluate(nn, boards, diagonals, true_values):
    ''' 
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
    '''

    total_boards = 0
    right_boards = 0
    mean_error = 0

    pred_value = tf.reshape(nn.pred_value, [-1])
    err = tf.sub(nn.true_value, pred_value)
    err_sum = tf.reduce_sum(err)

    guess_whos_winning = tf.equal(tf.round(nn.true_value), tf.round(pred_value))
    num_right = tf.reduce_sum(tf.cast(guess_whos_winning, tf.float32)) 

    for i in xrange(np.shape(boards)[0]):
        es, nr, gww, pv = nn.sess.run([err_sum, num_right, guess_whos_winning, pred_value], feed_dict={nn.data: boards[i], nn.data_diags: diagonals[i], nn.true_value: true_values[i]})
        total_boards += len(true_values[i])
        right_boards += nr
        mean_error += es

    mean_error = mean_error/total_boards
    print "mean_error: %f, guess who's winning correctly in %d out of %d games" % (mean_error, right_boards, total_boards)

