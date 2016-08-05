import sys
sys.path.insert(0, 'helpers/')

import tensorflow as tf
import numpy as np
import chess
import pickle
from chess_game_parser import get_fens
import stockfish_eval as sf

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


NUM_FEAT = 10
BATCH_SIZE = 5
NUM_HIDDEN = 1024
LEARNING_RATE = 0.001

NUM_CHANNELS = 6*2
piece_indices = {
    'p' : 0,
    'r' : 1,
    'n' : 2,
    'b' : 3,
    'q' : 4,
    'k' : 5,
}

def get_session():
    sess = tf.get_default_session()
    if sess is None:
        sess = tf.InteractiveSession()
    return sess

def neural_net(data, data_diags, true_value):

    sess = get_session()

    # weights
    W_grid = weight_variable([5,5,NUM_CHANNELS,NUM_FEAT])
    W_rank = weight_variable([8,1,NUM_CHANNELS,NUM_FEAT])
    W_file = weight_variable([1,8,NUM_CHANNELS,NUM_FEAT])
    W_diag = weight_variable([1,8,NUM_CHANNELS,NUM_FEAT])

    # biases
    b_grid = bias_variable([NUM_FEAT])
    b_rank = bias_variable([NUM_FEAT])
    b_file = bias_variable([NUM_FEAT])
    b_diag = bias_variable([NUM_FEAT])

    # fully connected layer 1, weights + biases
    W_fc_1 = weight_variable([90*NUM_FEAT, NUM_HIDDEN])
    b_fc_1 = bias_variable([NUM_HIDDEN])

    # fully connected layer 2, weights + biases
    W_fc_2 = weight_variable([NUM_HIDDEN, NUM_HIDDEN])
    b_fc_2 = bias_variable([NUM_HIDDEN])

    # Ouput layer
    W_final = weight_variable([NUM_HIDDEN,1])
    b_final = bias_variable([1])

    weights = {}
    weights['W_grid'] = W_grid
    weights['W_rank'] = W_rank
    weights['W_file'] = W_file
    weights['W_diag'] = W_diag

    weights['W_fc_1'] = W_fc_1
    weights['W_fc_2'] = W_fc_2
    weights['W_final'] = W_final

    weights['b_grid'] = b_grid
    weights['b_rank'] = b_rank
    weights['b_file'] = b_file
    weights['b_diag'] = b_diag

    weights['b_fc_1'] = b_fc_1
    weights['b_fc_2'] = b_fc_2
    weights['b_final'] = b_final

    sess.run(tf.initialize_all_variables())

    # outputs to conv layer
    o_grid = tf.nn.relu(conv5x5_grid(data, W_grid) + b_grid)
    o_rank = tf.nn.relu(conv8x1_line(data, W_rank) + b_rank)
    o_file = tf.nn.relu(conv8x1_line(data, W_file) + b_file)
    o_diag = tf.nn.relu(conv8x1_line(data_diags, W_diag) + b_diag)

    o_grid = tf.reshape(o_grid, [BATCH_SIZE, 64*NUM_FEAT])
    o_rank = tf.reshape(o_rank, [BATCH_SIZE, 8*NUM_FEAT])
    o_file = tf.reshape(o_file, [BATCH_SIZE, 8*NUM_FEAT])
    o_diag = tf.reshape(o_diag, [BATCH_SIZE, 10*NUM_FEAT])

    o_conn = tf.concat(1, [o_grid, o_rank, o_file, o_diag])

    # output of fully connected layer 1
    o_fc_1 = tf.nn.relu(tf.matmul(o_conn, W_fc_1) + b_fc_1)

    # output of fully connected layer 2
    o_fc_2 = tf.nn.relu(tf.matmul(o_fc_1, W_fc_2) + b_fc_2)

    # final_output
    pred_value = tf.sigmoid(tf.matmul(o_fc_2, W_final) + b_final)

    return pred_value, weights

def train(boards, diagonals, true_values, save_end_weights=True):

    sess = get_session()

    # placeholders
    data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,8,8,NUM_CHANNELS])
    data_diags = tf.placeholder(tf.float32, shape=[BATCH_SIZE,10,8,NUM_CHANNELS])
    true_value = tf.placeholder(tf.float32, shape=[BATCH_SIZE])

    pred_value, weights = neural_net(data, data_diags, true_value)
    # From my limited understanding x_entropy is not suitable - but if im wrong it could be better
    # Using squared error instead
    cost = tf.reduce_sum(tf.pow(tf.sub(pred_value, true_value), 2))

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)

    for i in xrange(np.shape(boards)[0]):  
        sess.run([train_step], feed_dict={data: boards[i], data_diags: diagonals[i], true_value: true_values[i]})

    if save_end_weights:
        weight_values = {}
        weight_values['W_grid'] = sess.run(weights['W_grid'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['W_rank'] = sess.run(weights['W_rank'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['W_file'] = sess.run(weights['W_file'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['W_diag'] = sess.run(weights['W_diag'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})

        weight_values['W_fc_1'] = sess.run(weights['W_fc_1'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['W_fc_2'] = sess.run(weights['W_fc_2'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['W_final'] = sess.run(weights['W_final'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})

        weight_values['b_grid'] = sess.run(weights['b_grid'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['b_rank'] = sess.run(weights['b_rank'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['b_file'] = sess.run(weights['b_file'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['b_diag'] = sess.run(weights['b_diag'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})

        weight_values['b_fc_1'] = sess.run(weights['b_fc_1'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['b_fc_2'] = sess.run(weights['b_fc_2'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})
        weight_values['b_final'] = sess.run(weights['b_final'], feed_dict={data: boards[0], data_diags: diagonals[0], true_value: true_values[0]})

        pickle.dump(weight_values, open('weight_values.p', 'wb'))

def evaluate(boards, diagonals, true_values):

    total_boards = 0
    right_boards = 0
    mean_error = 0

    sess = get_session()

    # placeholders
    data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,8,8,NUM_CHANNELS])
    data_diags = tf.placeholder(tf.float32, shape=[BATCH_SIZE,10,8,NUM_CHANNELS])
    true_value = tf.placeholder(tf.float32, shape=[BATCH_SIZE])

    pred_value, weights = neural_net(data, data_diags, true_value)

    pred_value = tf.reshape(pred_value, [-1])

    err = tf.sub(true_value, pred_value)

    err_sum = tf.reduce_sum(err)

    guess_whos_winning = tf.equal(tf.round(true_value), tf.round(pred_value))
    num_right = tf.reduce_sum(tf.cast(guess_whos_winning, tf.float32)) 

    for i in xrange(np.shape(boards)[0]):
        es, nr, gww, pv = sess.run([err_sum, num_right, guess_whos_winning, pred_value], feed_dict={data: boards[i], data_diags: diagonals[i], true_value: true_values[i]})
        print pv
        print true_values[i]
        total_boards += len(true_values[i])
        right_boards += nr
        mean_error += es

    mean_error = mean_error/total_boards
    print "mean_error: %f, guess who's winning correctly in %d out of %d games" % (mean_error, right_boards, total_boards)


    ## FEED THE PLACEHOLDER'S THEY'RE HUNGRY
def load_weight_values(filename = 'weight_values.p'):
    weight_values = pickle.load(open(filename, 'rb'))
    return weight_values

def load_stockfish_values(filename = 'true_values.p'):
    stockfish_values = pickle.load(open(filename, 'rb'))
    return stockfish_values

def fen_to_channels(fen):
    """
    Converts a fen string to channels for neural net.
    Always assumes that it's white's turn
    @TODO deal with en passant and castling

    Inputs:
        epd:
            epd or fen string describing current state. Currently only using board state

    Output:
        Channels:

        Consists of 3 6x8x8 channels (6 8x8 chess boards)
        The three channels are:
        1. your pieces
        2. opponents pieces
        3. all pieces

        Each channel has 6 boards, for each of the 6 types of pieces.
        In order they are Pawns, Rooks, Knights, Bishops, Queens, Kings.
    """

    # fen = fen.split(' ')
    # board_str = fen[0]
    # turn = fen[1]
    # castling = fen[2]
    # en_passant = fen[3]

    channels = np.zeros((8,8,NUM_CHANNELS))

    file = 0
    rank = 0
    empty_char = False
    for char in fen:
        if char == '/':
            file = 0
            rank += 1
            continue
        elif char.isdigit():
            file += int(char)
            continue
        else:
            my_piece = char.islower() # double check this. Normal fen, black is lower, but stockfish seems use to lower as current move
            char = char.lower()
            if my_piece:
                channels[rank, file, piece_indices[char]] = 1
            else:
                channels[rank, file, piece_indices[char] + 6] = 1

            # channels[rank, file, piece_indices[char] + 12] = 1 if my_piece else -1
        file += 1
        if rank == 7 and file == 8:
            break
    return channels


def get_diagonals(channels):
    """
    Retrieves and returns the diagonals from the board

    Ouput:
        3 Channels: your pieces, opponents pieces, all pieces
        Each channel has 6 arrays for each piece, in order: Pawns, Rooks, Knights, Bishops, Queens, King
        Each piece array has 10 diagonals with max size of 8 (shorter diagonasl are 0 padded)
    """
    diagonals = np.zeros((10, 8, NUM_CHANNELS))
    for piece_idx in piece_indices.values():

        # diagonals with length 6 and 7
        for length in xrange(6,8):
            for i in xrange(length):
                offset = 8-length
                diag_offset = 4 if length == 7 else 0
                for channel in xrange(NUM_CHANNELS):
                    # upwards diagonals
                    diagonals[0+diag_offset, int(offset/2)+i, channel] = channels[i+offset, i, channel]
                    diagonals[1+diag_offset, int(offset/2)+i, channel] = channels[i, i+offset, channel]
                    #downwards diagonals
                    diagonals[2+diag_offset, int(offset/2)+i, channel] = channels[7-offset-i, i, channel]
                    diagonals[3+diag_offset, int(offset/2)+i, channel] = channels[7-i, offset-i, channel]

        # diagonals with length 8
        for i in xrange(8):
            for channel in xrange(NUM_CHANNELS):
                # upwards
                diagonals[8, i, channel] = channels[i, i, channel]
                # downwards
                diagonals[9, i, channel] = channels[7-i, i, channel]

    return diagonals


def get_stockfish_values(boards):
    ''' Uses stockfishes evaluation to get a score for each board, then uses a sigmoid to map
        the scores to a winning probability between 0 and 1 (see sigmoid_array for how the sigmoid was chosen)

        inputs:
            boards:
                list of board fens

        outputs:
            values:
                a list of values for each board ranging between 0 and 1
    '''        
    cps = []
    i = 0
    for b in boards:
    # cp = centipawns advantage
        cp = sf.stockfish_scores(b, seconds=2)
        print cp
        if cp is not None:
            cps.append(cp)
    cps = np.array(cps)
    print np.shape(cps)
    return sigmoid_array(cps)

def sigmoid_array(values):
    ''' From: http://chesscomputer.tumblr.com/post/98632536555/using-the-stockfish-position-evaluation-score-to
        1000 cp lead almost guarantees a win (a sigmoid within that). From the looking at the graph to gather a few data point
        and using a sigmoid curve fitter an inaccurate function of 1/(1+e^(-0.00547x)) was decided on (by me, deal with it)
        Ideally this fitter function is learned, but this is just for testing so...'''
    return 1./(1. + np.exp(-0.00547*values))


load_true_values = True
raw_input('This will overwrite your old weights\' pickle, do you still want to proceed? (Hit Enter)')

print 'Training data. Will save weights to pickle'

fens = get_fens(num_games=1)[:10]
print "Finished retrieving %d fens.\nBegin retrieving stockfish values.\n" % (len(fens))

num_batches = len(fens)/BATCH_SIZE

boards = np.zeros((num_batches, BATCH_SIZE, 8, 8, NUM_CHANNELS))
diagonals = np.zeros((num_batches, BATCH_SIZE, 10, 8, NUM_CHANNELS))

if load_true_values:
    true_values = load_stockfish_values()
else:
    true_values = get_stockfish_values(fens[:num_batches*BATCH_SIZE])
    # save stockfish_values
    pickle.dump(true_values, open('true_values.p', 'wb'))

true_values = np.reshape(true_values, (num_batches, BATCH_SIZE))
print "Finished getting stockfish values. Begin training neural_net with %d items" % (len(fens))

for i in xrange(num_batches*BATCH_SIZE):
    batch_num = i/BATCH_SIZE
    batch_idx = i % BATCH_SIZE
    boards[batch_num][batch_idx] = fen_to_channels(fens[i])

    for i in xrange(BATCH_SIZE):
        diagonals[batch_num][batch_idx] = get_diagonals(boards[batch_num][batch_idx])

# # print channels[0][0][0]

train(boards, diagonals, true_values)
# print true_values
evaluate(boards, diagonals, true_values)
# for i in xrange(1,10):
# print sf.stockfish_scores(board.fen().split(' ')[0], seconds = 1)
# print sf.stockfish_scores("4R1K1/PPP1NR2/3Q2P1/3p4/3nk1p1/1q1p3p/pp1b2b1/rn5r", seconds = 1) #  mate in 2 if whtie

