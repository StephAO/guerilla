import tensorflow as tf
import numpy as np
import chess
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
BATCH_SIZE = 100
NUM_HIDDEN = 1024

NUM_CHANNELS = 6*2
piece_indices = {
    'p' : 0,
    'r' : 1,
    'n' : 2,
    'b' : 3,
    'q' : 4,
    'k' : 5,
}

def neural_net(channels):

    diagonals = np.zeros((BATCH_SIZE,10,8,NUM_CHANNELS))
    for i in xrange(BATCH_SIZE):
        diagonals[i] = get_diagonals(channels[i])

    sess = tf.InteractiveSession()
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
    W_final = weight_variable([NUM_HIDDEN])
    b_final = bias_variable([1])

    # placeholders
    data = tf.placeholder(tf.float32, shape=[BATCH_SIZE,8,8,NUM_CHANNELS])
    data_diags = tf.placeholder(tf.float32, shape=[BATCH_SIZE,10,8,NUM_CHANNELS])
    real_value = tf.placeholder(tf.float32, shape=[BATCH_SIZE])

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
    pred_value = tf.tanh(tf.matmul(o_fc_2, W) + b)

    log_l = (-1)*(pred_value)

    train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(log_l)

    # a,b,c,d = sess.run([o_grid, o_rank, o_file, o_diag], feed_dict={data: channels, data_diags: diagonals})
    # print np.shape(a)
    # print np.shape(b)
    # print np.shape(c)
    # print np.shape(d)


    print np.shape(o_grid.eval(feed_dict={data: channels, data_diags: diagonals}))
    print np.shape(o_rank.eval(feed_dict={data: channels, data_diags: diagonals}))
    print np.shape(o_file.eval(feed_dict={data: channels, data_diags: diagonals}))
    print np.shape(o_diag.eval(feed_dict={data: channels, data_diags: diagonals}))
    print np.shape(o_conn.eval(feed_dict={data: channels, data_diags: diagonals}))

    print np.shape(o_fc_1.eval(feed_dict={data: channels, data_diags: diagonals}))

    ## FEED THE PLACEHOLDER'S THEY'RE HUNGRY



def epd_to_channels(epd):
    """
    Converts and epd string to channels for neural net.
    @TODO deal with en passant and castling

    Inputs:
        epd:
            epd or fen string describing current state (no distinction has been made yet)

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

    epd = epd.split(' ')
    board_str = epd[0]
    turn = epd[1]
    castling = epd[2]
    en_passant = epd[3]

    channels = np.zeros((8,8,NUM_CHANNELS))

    file = 0
    rank = 0
    empty_char = False
    for char in board_str:
        if char == '/':
            file = 0
            rank += 1
            continue
        elif char.isdigit():
            file += int(char)
            continue
        else:
            my_piece = (char.islower() and turn == 'w') or (char.isupper() and turn == 'b')
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






board = chess.Board(chess.STARTING_FEN, chess960=False)
print board.epd()
# channels = np.zeros((BATCH_SIZE,8,8,NUM_CHANNELS))
# channels[0] = epd_to_channels(board.epd())

# print channels[0][0][0]

# neural_net(channels)
# for i in xrange(1,10):
print sf.stockfish_scores(board.fen().split(' ')[0], seconds = 1) # "4R1K1/PPP1NR2/3Q2P1/3p4/3nk1p1/1q1p3p/pp1b2b1/rn5r" mate in 2 if whtie

