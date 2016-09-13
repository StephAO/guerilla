import os
import random
import numpy as np
import tensorflow as tf
import math
import chess.pgn
import chess

import guerilla
import data_handler as dh
import stockfish_eval as sf
import chess_game_parser as cgp
from hyper_parameters import *

class Teacher:
    def __init__(self, _guerilla, actions):

        # dictionary of different training/evaluation methods
        self.actions_dict = {
            'train_bootstrap' : self.train_bootstrap,
            'train_td_endgames' : self.train_td_endgames,
        }

        self.guerilla = _guerilla
        self.nn = _guerilla.nn
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.actions = actions

        # Bootstrap parameters
        self.fens_filename = "fens.p"
        self.stockfish_filename = "sf_scores.p"

        # TD-Leaf parameters
        self.td_pgn_folder = self.dir_path + '/../helpers/pgn_files/single_game_pgns'
        self.td_rand_file = False  # If true then TD-Leaf randomizes across the files in the folder.
        self.td_num_endgame = -1  # The number of endgames to train on using TD-Leaf (-1 = All)
        self.td_num_full = -1  # The number of full games to train on using TD-Leaf
        self.td_depth = 12  # How many moves are included in each TD training

    def run(self):
        """ 
            1. load data from file
            2. configure data
            3. run actions
        """

        for action in self.actions:
            if action == 'train_bootstrap':
                print "Performing Bootstrap training!"
                print "Fetching stockfish values..."

                # Fetch stockfish values
                fens = cgp.load_fens(self.fens_filename)

                num_batches = len(fens) / BATCH_SIZE

                true_values = sf.load_stockfish_values(self.stockfish_filename)
                true_values = np.reshape(true_values[:num_batches * BATCH_SIZE], (num_batches, BATCH_SIZE))

                print "Finished getting stockfish values. Begin training neural_net with %d items" % (len(fens))

                boards = np.zeros((num_batches, BATCH_SIZE, 8, 8, NUM_CHANNELS))
                diagonals = np.zeros((num_batches, BATCH_SIZE, 10, 8, NUM_CHANNELS))

                self.train_bootstrap(boards, diagonals, true_values, num_batches, fens)

            elif action == 'train_td_endgames':
                print "Performing endgame TD-Leaf training!"
                self.train_td_endgames()

            else:
                raise NotImplementedError, "Error: %s is not a valid action." % (action)

    def train_bootstrap(self, boards, diagonals, true_values, num_batches, fens, save_weights=True):
        """
            train neural net
        """

        # train should depend on action
        raw_input('This will overwrite your old weights\' pickle, do you still want to proceed? (Hit Enter)')
        print 'Training data. Will save weights to pickle'

        num_boards = num_batches * BATCH_SIZE

        for epoch in xrange(NUM_EPOCHS):
            # Configure data (shuffle fens -> fens to channel -> group batches)
            game_indices = range(num_boards)
            random.shuffle(game_indices)
            for i in range(num_boards):
                batch_num = i / BATCH_SIZE
                batch_idx = i % BATCH_SIZE

                boards[batch_num][batch_idx] = dh.fen_to_channels(fens[game_indices[i]])
                for j in xrange(BATCH_SIZE):  # Maybe use matrices instead of for loop for speed
                    diagonals[batch_num][batch_idx] = dh.get_diagonals(boards[batch_num][batch_idx])
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


    def set_td_params(self, num_end=None, num_full=None, randomize=None, pgn_folder=None, depth = None):
        """
        Set the parameters for TD-Leaf.
            Inputs:
                num_end [Int]
                    Number of endgames to train on using TD-Leaf.
                num_full [Int]
                    Number of full games to train on using TD-Leaf.
                randomize [Boolean]
                    Whether or not to randomize across the pgn files.
                pgn_folder [String]
                    Folder containing chess games in PGN format.
        """

        if num_end:
            self.td_num_endgame = num_end
        if num_full:
            self.td_num_full = num_full
        if randomize:
            self.td_rand_file = randomize
        if pgn_folder:
            self.td_pgn_folder = pgn_folder
        if depth:
            self.td_depth = depth

    def train_td_endgames(self):
        """
        Trains the neural net using TD-Leaf on endgames.
        """

        game_indices = range(self.td_num_endgame)

        if self.td_rand_file:
            random.shuffle(game_indices)
            pgn_files = [f for f in os.listdir(self.td_pgn_folder) if os.path.isfile(os.path.join(self.td_pgn_folder, f))]
        else:
            pgn_files = [f for f in os.listdir(self.td_pgn_folder)[:self.td_num_endgame] if
                         os.path.isfile(os.path.join(self.td_pgn_folder, f))]

        for i, game_idx in enumerate(game_indices):
            print "Training on game %d of %d..." % (i + 1, self.td_num_endgame)
            fens = []

            # Open and use pgn file sequentially or at random
            with open(os.path.join(self.td_pgn_folder, pgn_files[game_idx])) as pgn:
                game = chess.pgn.read_game(pgn)

                # Only get endgame fens
                curr_node = game.end()
                for i in range(self.td_depth):
                    fens.insert(0, curr_node.board().fen())

                    # Check if start of game is reached
                    if curr_node == game.root():
                        break
                    curr_node = curr_node.parent

            # Call TD-Leaf
            # for i in range(len(fens)):
            #     print self.nn.evaluate(fens[i]) if i%2==0 else 1 - self.nn.evaluate(dh.flip_board(fens[i]))
            self.td_leaf(fens)

    def td_leaf(self, game):
        """
        Trains neural net using TD-Leaf algorithm.

            Inputs:
                Game [List]
                    A game consists of a sequential list of board states. Each board state is a FEN.
        """
        # TODO: Maybe this should check that each game is valid? i.e. assert that only legal moves are played.
        # TODO: Add LEAF part of TD-Leaf

        num_boards = len(game)
        game_info = [{'board': None, 'value': None} for _ in range(num_boards)]  # Indexed the same as num_boards
        w_update = None

        # Pre-calculate leaf value (J_d(x,w)) of search applied to each board
        # Get new board state from leaf
        # Note: Does not modify score of black boards.
        for i, board in enumerate(game):
            game_info[i]['value'], _, game_info[i]['board'] = self.guerilla.search.run(chess.Board(board))

        print "TD-Leaf values calculated!"

        for t in range(num_boards):
            td_val = 0
            for j in range(t, num_boards - 1):
                # Calculate temporal difference
                dt = self.calc_value_diff(game_info[j], game_info[j + 1])
                # Add to sum
                td_val += math.pow(TD_DISCOUNT, j - t)*dt

            # Get gradient and update sum
            update = self.nn.get_gradient(game_info[j]['board'], self.nn.all_weights)
            if not w_update:
                w_update = [w*td_val for w in update]
            else:
                # update each set of weights
                for i in range(len(update)):
                    w_update[i] += update[i] * td_val

        # Update neural net weights.
        self.nn.update_weights(self.nn.all_weights, [w*TD_LRN_RATE for w in w_update])

    def calc_value_diff(self, curr_board, next_board):
        """
        Calculates the score difference between two board states.
            Inputs:
                curr_board [Dict]
                    {'board': FEN of current board state, 'value': value of current board state}
                next_board [Dict]
                    {'board': FEN of next board state, 'value': value of next board state}
            Output:
                score_diff [Float]
                    Value difference.
        """
        assert ((dh.fen_is_white(curr_board['board']) and dh.fen_is_black(next_board['board'])) or
                (dh.fen_is_black(curr_board['board']) and dh.fen_is_white(next_board['board'])))

        if dh.fen_is_black(curr_board['board']):
            score_diff = (next_board['value']) - (1 - curr_board['value'])
        else:
            score_diff = (1 - next_board['value']) - (curr_board['value'])

        return score_diff

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
    g = guerilla.Guerilla('Harambe', 'w','weight_values.p')
    t = Teacher(g, ['train_td_endgames'])#['train_bootstrap'])
    t.set_td_params(num_end=1, num_full=1,randomize=False)
    t.run()


if __name__ == '__main__':
    main()