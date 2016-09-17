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
            'train_bootstrap': self.train_bootstrap,
            'train_td_endgames': self.train_td,
            'train_td_full': self.train_td,
            'load_and_resume' : self.resume
        }

        self.guerilla = _guerilla
        self.nn = _guerilla.nn
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.actions = actions
        self.start_time = None
        self.training_time = None
        self.files = None

	# TD-Leaf parameters
        self.td_pgn_folder = self.dir_path + '/../helpers/pgn_files/single_game_pgns'
        self.td_rand_file = False  # If true then TD-Leaf randomizes across the files in the folder.
        self.td_num_endgame = -1  # The number of endgames to train on using TD-Leaf (-1 = All)
        self.td_num_full = -1  # The number of full games to train on using TD-Leaf
        self.td_depth = 12  # How many moves are included in each TD training

    def run(self, training_time = None, fens_filename = "fens.p", stockfish_filename = "sf_scores.p"):
        """ 
            1. load data from file
            2. configure data
            3. run actions
        """
	self.files = [fens_filename, stockfish_filename]
	self.start_time = time.time()
        self.training_time = training_time
       
	for action in self.actions:
            if action == 'train_bootstrap':
		print "Performing Bootstrap training!"
                print "Fetching stockfish values..."

		fens = cgp.load_fens(fens_filename)
		fens = fens[:(-1) * (len(fens) % BATCH_SIZE)]
		true_values = sf.load_stockfish_values(stockfish_filename)[:len(fens)]
		
		self.train_bootstrap(fens, true_values)
        	
        for action in self.actions:
            if action in self.actions_dict:
                self.actions_dict[action](boards, diagonals, true_values, num_batches, fens)
            elif action == 'train_td_endgames':
                print "Performing endgame TD-Leaf training!"
                self.train_td(endgame=True)
            elif action == 'train_td_full':
                print "Performing full-game TD-Leaf training!"
                self.train_td(endgame=False)
            else:
                raise NotImplementedError("Error: %s is not a valid action." % action)

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
        self.nn.save_weight_values(filename = 'in_training_weight_values.p')
        pickle.dump(state, open(pickle_path, 'wb'))

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
        print "Resuming training"
        state = self.load_state()
        fens = cgp.load_fens(self.files[0])

        num_batches = len(fens) / BATCH_SIZE

        true_values = sf.load_stockfish_values(self.files[1])[:len(fens)]

        self.start_time = time.time()
        self.training_time = training_time;

        if state['action'] == 'train_bootstrap':
            # finish epoch
            train_fens = fens[:(-1) * CONV_CHECK_SIZE] # fens to train on
            convg_fens = fens[(-1) * CONV_CHECK_SIZE:] # fens to check convergence on

            train_values = true_values[:(-1) * CONV_CHECK_SIZE]
            convg_values = true_values[(-1) * CONV_CHECK_SIZE:]
            cost = tf.reduce_sum(tf.pow(tf.sub(self.nn.pred_value, self.nn.true_value), 2))
            train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
            self.weight_update_bootstrap(train_fens, train_values, state['remaining_boards'], cost, train_step)

            # evaluate nn for convergence
            state['error'].append(self.evaluate_bootstrap(convg_fens, convg_values))
            base_loss = state['error'][0] - state['error'][1]
            curr_loss = state['error'][-2] - state['error'][-1]
            if (abs(curr_loss / base_loss) < CONV_THRESHOLD):
                self.nn.save_weight_values()
                plt.plot(range(state['epoch_num']), state['error'])
                plt.show()
                return

            # continue with rests of epochs
            self.train_bootstrap(fens, true_values, start_epoch=state['epoch_num'], error=state['error'])

    def train_bootstrap(self, fens, true_values, start_epoch = 0, error = []):
        """
            Train neural net

            Inputs:
                fens[list of strings]:
                    The fens representing the board states
                true_values[ndarray]:
                    Expected output for each chess board state (between 0 and 1)
                num_batches[int]:
                    number of batches
        """

        train_fens = fens[:(-1) * CONV_CHECK_SIZE] # fens to train on
        convg_fens = fens[(-1) * CONV_CHECK_SIZE:] # fens to check convergence on

        train_values = true_values[:(-1) * CONV_CHECK_SIZE]
        convg_values = true_values[(-1) * CONV_CHECK_SIZE:]

        num_boards = len(train_fens)

        usr_in = raw_input("This will overwrite your old weights\' pickle, do you still want to proceed (y/n)?: ")
        if usr_in.lower() == 'n':
            return
        print "Training data on %d positions. Will save weights to pickle" % (num_boards)

        # From my limited understanding x_entropy is not suitable - but if im wrong it could be better
        # Using squared error instead
        cost = tf.reduce_sum(tf.pow(tf.sub(self.nn.pred_value, self.nn.true_value), 2))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
        epoch = start_epoch-1
        for epoch in xrange(start_epoch, NUM_EPOCHS):
            print epoch
            # Configure data (shuffle fens -> fens to channel -> group batches)
            game_indices = range(num_boards)
            random.shuffle(game_indices)

            # update weights
            save = self.weight_update_bootstrap(train_fens, train_values, game_indices, cost, train_step)

            # save state if timeout
            if save[0]:
                save[1]['epoch_num'] = epoch + 1
                save[1]['error'] = error
                self.save_state(save[1])
                return

            # evaluate nn for convergence
            error.append(self.evaluate_bootstrap(convg_fens, convg_values))
            print error[-1]
            if (len(error) > 2):
                base_loss = error[0] - error[1]
                curr_loss = error[-2] - error[-1]
                if (abs(curr_loss / base_loss) < CONV_THRESHOLD):
                    print "Training complete: Reached convergence threshold"
                    break

        else:
            print "Training complete: Reached max epoch, no convergence yet"

        self.nn.save_weight_values()

        plt.plot(range(epoch+1), error)
        plt.show()
        return

    def weight_update_bootstrap(self, fens, true_values_, game_indices, cost, train_step):
        """ Weight update for a single batch """        

        if len(game_indices) % BATCH_SIZE != 0:
            raise Exception("Error: number of fens provided (%d) is not a multiple of batch_size (%d)" % (len(game_indices), BATCH_SIZE))

        num_batches = int(len(game_indices) / BATCH_SIZE)

        board_num = 0
        boards = np.zeros((BATCH_SIZE, 8, 8, NUM_CHANNELS))
        diagonals = np.zeros((BATCH_SIZE, 10, 8, NUM_CHANNELS))
        true_values = np.zeros((BATCH_SIZE))

        for i in xrange(num_batches):
            # if training time is up, save state
            if self.training_time is not None \
                and time.time() - self.start_time >= self.training_time:
                print "Timeout: Saving state and quitting"
                return (True, 
                        {   
                            'remaining_boards'   : game_indices[(i * BATCH_SIZE):],
                            'action'             : 'train_bootstrap'
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


    def evaluate_bootstrap(self, fens, true_values):
        """
            Evaluate neural net

            Inputs:
                fens[list of strings]:
                    The fens representing the board states
                true_values[ndarray]:
                    Expected output for each chess board state (between 0 and 1)
        """

        total_boards = 0
        right_boards = 0
        mean_error = 0

        # Create tensors
        pred_value = tf.reshape(self.nn.pred_value, [-1])
        err = tf.sub(self.nn.true_value, pred_value)
        err_sum = tf.reduce_sum(err)

        # Configure data
        boards = np.zeros((CONV_CHECK_SIZE, 8, 8, NUM_CHANNELS))
        diagonals = np.zeros((CONV_CHECK_SIZE, 10, 8, NUM_CHANNELS))
        for i in xrange(CONV_CHECK_SIZE):
            boards[i] = dc.fen_to_channels(fens[i])
            diagonals[i] = dc.get_diagonals(boards[i])

        # Get loss
        error = self.nn.sess.run([err_sum], feed_dict = { 
                self.nn.data: boards, 
                self.nn.data_diags: diagonals,
                self.nn.true_value: true_values 
        })

        return abs(error[0])

    def set_td_params(self, num_end=None, num_full=None, randomize=None, pgn_folder=None, depth=None):
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

    def train_td(self, endgame):
        """
        Trains the neural net using TD-Leaf.
            Inputs:
                endgame [Boolean]
                    If True then only trains on endgames. Otherwise trains on a random subset of full games.
        """

        num_games = self.td_num_endgame if endgame else self.td_num_full

        # Only load some files if not random
        if self.td_rand_file:
            pgn_files = [f for f in os.listdir(self.td_pgn_folder) if
                         os.path.isfile(os.path.join(self.td_pgn_folder, f))]
        else:
            pgn_files = [f for f in os.listdir(self.td_pgn_folder)[:num_games] if
                         os.path.isfile(os.path.join(self.td_pgn_folder, f))]

        game_indices = range(num_games if num_games >= 0 else len(pgn_files))

        # Shuffle if necessary
        if self.td_rand_file:
            random.shuffle(game_indices)

        for i, game_idx in enumerate(game_indices):
            print "Training on game %d of %d..." % (i + 1, num_games)
            fens = []

            # Open and use pgn file sequentially or at random
            with open(os.path.join(self.td_pgn_folder, pgn_files[game_idx])) as pgn:
                game = chess.pgn.read_game(pgn)

                if endgame:
                    # Only get endgame fens
                    curr_node = game.end()
                    for _ in range(self.td_depth):
                        fens.insert(0, curr_node.board().fen())

                        # Check if start of game is reached
                        if curr_node == game.root():
                            break
                        curr_node = curr_node.parent
                else:
                    # Get all fens
                    curr_node = game.root()
                    while True:
                        fens.append(curr_node.board().fen())

                        # Check if end has been reached
                        if curr_node.is_end():
                            break

                        curr_node = curr_node.variations[0]

                    # Get random subset
                    game_length = len(fens)
                    sub_start = random.randint(0, max(0, game_length - self.td_depth))
                    sub_end = min(game_length, sub_start + self.td_depth)
                    fens = fens[sub_start:sub_end]

                    # TODO: Remove this checklater.
                    if (len(fens) != self.td_depth and game_length >= self.td_depth) or \
                            (len(fens) != game_length and game_length < self.td_depth):
                        print "Warning: This shouldn't happen!"

            # Call TD-Leaf
            self.td_leaf(fens)

    def td_leaf(self, game):
        """
        Trains neural net using TD-Leaf algorithm.

            Inputs:
                Game [List]
                    A game consists of a sequential list of board states. Each board state is a FEN.
        """
        # TODO: Maybe this should check that each game is valid? i.e. assert that only legal moves are played.

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
                td_val += math.pow(TD_DISCOUNT, j - t) * dt

            # Get gradient and update sum
            update = self.nn.get_gradient(game_info[t]['board'], self.nn.all_weights)
            if not w_update:
                w_update = [w * td_val for w in update]
            else:
                # update each set of weights
                for i in range(len(update)):
                    w_update[i] += update[i] * td_val

        # Update neural net weights.
        old_weights = self.nn.get_weights(self.nn.all_weights)
        new_weights = [old_weights[i] + TD_LRN_RATE * w_update[i] for i in range(len(w_update))]
        self.nn.update_weights(self.nn.all_weights, new_weights)
        print "Weights updated."

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

def main():
    g = guerilla.Guerilla('Harambe', 'w', 'weight_values.p')
    t = Teacher(g, ['train_bootstrap', 'train_td_endgames', 'train_td_full'])
    t.set_td_params(num_end=2, num_full=1, randomize=False)
    # t.run(training_time = 3600, fens_filename = "fens_1000.p", stockfish_filename = "true_values_1000.p")


if __name__ == '__main__':
    main()