import os
import random
import numpy as np
import tensorflow as tf
import math
import chess.pgn
import chess
import time
import matplotlib.pyplot as plt
import pickle

import guerilla
import data_handler as dh
import stockfish_eval as sf
import chess_game_parser as cgp
from hyper_parameters import *

##### just for testing memory #####
from guppy import hpy
###################################

class Teacher:

    def __init__(self, _guerilla):

        # dictionary of different training/evaluation methods
        # TODO: Shouldn't this be a static class variable?
        self.actions_dict = [
            'train_bootstrap',
            'train_td_endgames',
            'train_td_full',
            'load_and_resume',
            'train_selfplay'
        ]

        self.guerilla = _guerilla
        self.nn = _guerilla.nn
        self.dir_path = os.path.dirname(os.path.abspath(__file__))
        self.start_time = None
        self.training_time = None
        self.files = None
        self.actions = None
        self.curr_action_idx = None
        self.saved = None

        # TD-Leaf parameters
        self.td_pgn_folder = self.dir_path + '/../helpers/pgn_files/single_game_pgns'
        self.td_rand_file = False  # If true then TD-Leaf randomizes across the files in the folder.
        self.td_num_endgame = -1  # The number of endgames to train on using TD-Leaf (-1 = All)
        self.td_num_full = -1  # The number of full games to train on using TD-Leaf
        self.td_end_length = 12  # How many moves are included in endgame training
        self.td_full_length = -1  # Maximum number of moves for full game training (-1 = All)

        # Self-play parameters
        self.sp_num = 1  # The number of games to play against itself
        self.sp_length = 12  # How many moves are included in game playing

    # ---------- RUNNING AND RESUMING METHODS

    def run(self, actions, training_time=None, fens_filename="fens.p", stockfish_filename="sf_scores.p"):
        """ 
            1. load data from file
            2. configure data
            3. run actions
        """
        self.files = [fens_filename, stockfish_filename]
        self.start_time = time.time()
        self.training_time = training_time
        self.actions = actions
        self.curr_action_idx = 0  # This value gets modified if resuming
        self.saved = False
        weight_values = None

        if self.actions[0] == 'load_and_resume':
            weight_values = self.resume(training_time)

        # Note: This cannot be a for loop as self.curr_action_idx gets set to non-zero when resuming.
        while True:
            # Check if done
            if self.curr_action_idx >= len(self.actions):
                break
            elif self.out_of_time():
                # Check if run out of time between actions, save if necessary
                if not self.saved:
                    self.save_state(state={})
                break

            # Else
            action = self.actions[self.curr_action_idx]
            if action == 'train_bootstrap':
                print "Performing Bootstrap training!"
                print "Fetching stockfish values..."

                fens = cgp.load_fens(fens_filename)
                if (len(fens) % BATCH_SIZE) != 0:
                    fens = fens[:(-1) * (len(fens) % BATCH_SIZE)]
                true_values = sf.load_stockfish_values(stockfish_filename)[:len(fens)]

                weight_values = self.train_bootstrap(fens, true_values, weight_values=weight_values)
            elif action == 'train_td_endgames':
                print "Performing endgame TD-Leaf training!"
                weight_values = self.train_td(weight_values, True)
            elif action == 'train_td_full':
                print "Performing full-game TD-Leaf training!"
                weight_values = self.train_td(weight_values, False)
            elif action == 'train_selfplay':
                print "Performing self-play training!"
                weight_values = self.train_selfplay(weight_values)
            elif action == 'load_and_resume':
                raise ValueError("Error: Resuming must be the first action in an action set.")
            else:
                raise NotImplementedError("Error: %s is not a valid action." % action)

            self.curr_action_idx += 1

            # Save new weight values
            
            if not self.saved:
                weight_file = "weights_" + action + "_" + time.strftime("%Y%m%d-%H%M%S") +".p"
                self.nn.save_weight_values(_filename=weight_file)
                print "Weights saved to %s" % weight_file

    def save_state(self, state, filename="state.p"):
        """
            Save current state so that training can resume at another time
            Note: Can only save state at whole batch intervals (but mid-epoch is fine)
            Inputs:
                state[dict]:
                    For bootstrap:
                        game_indices[list of ints]
                        (if action == 'train_bootstrap')
                            epoch_num[int]
                            loss[list of floats]
                        (if action == 'train_td_...')
                            start_idx [int]
                filename[string]:
                    filename to save pickle to
        """
        state['files'] = self.files

        state['curr_action_idx'] = self.curr_action_idx
        state['actions'] = self.actions

        # Save training parameters
        state['td_leaf_param'] = {'randomize': self.td_rand_file,
                                  'num_end': self.td_num_endgame,
                                  'num_full': self.td_num_full,
                                  'end_length': self.td_end_length,
                                  'full_length': self.td_full_length}
        state['sp_param'] = {'num_selfplay': self.sp_num, 'max_length': self.sp_length}

        pickle_path = self.dir_path + '/../pickles/' + filename
        self.nn.save_weight_values(_filename='in_training_weight_values.p')
        pickle.dump(state, open(pickle_path, 'wb'))
        self.saved = True
        print "State saved."

    def load_state(self, filename='state.p'):
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

        # Load training parameters
        self.set_td_params(**state.pop('td_leaf_param'))
        self.set_sp_params(**state.pop('sp_param'))

        self.files = state['files']
        self.curr_action_idx = state['curr_action_idx']
        self.actions = state['actions'] + self.actions[1:]
        # TODO this will called shortly after already loading weight values, can we remove the unecessary work
        self.nn.load_weight_values(_filename='in_training_weight_values.p') 
        return state

    def resume(self, training_time=None):
        """
            Resumes training from a previously paused training session
        """
        print "Resuming training"
        state = self.load_state()

        self.start_time = time.time()
        self.training_time = training_time

        if 'game_indices' not in state:
            # Stopped between actions.
            return

        action = self.actions[self.curr_action_idx]
        weight_values = None

        if action == 'train_bootstrap':
            print "Resuming Bootstrap training..."

            fens = cgp.load_fens(self.files[0])
            if (len(fens) % BATCH_SIZE) != 0:
                fens = fens[:(-1) * (len(fens) % BATCH_SIZE)]

            true_values = sf.load_stockfish_values(self.files[1])[:len(fens)]
            # finish epoch
            train_fens = fens[:(-1) * VALIDATION_SIZE]  # fens to train on
            valid_fens = fens[(-1) * VALIDATION_SIZE:]  # fens to check convergence on

            train_values = true_values[:(-1) * VALIDATION_SIZE]
            valid_values = true_values[(-1) * VALIDATION_SIZE:]
            cost = tf.reduce_sum(tf.pow(tf.sub(self.nn.pred_value, self.nn.true_value), 2))
            train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
            self.weight_update_bootstrap(train_fens, train_values, state['game_indices'], train_step)

            # evaluate nn for convergence
            state['error'].append(self.evaluate_bootstrap(valid_fens, valid_values))
            base_loss = state['error'][0] - state['error'][1]
            curr_loss = state['error'][-2] - state['error'][-1]
            if abs(curr_loss / base_loss) < LOSS_THRESHOLD:
                self.nn.save_weight_values()
                plt.plot(range(state['epoch_num']), state['error'])
                plt.show()

            # continue with rests of epochs
            weight_values = self.train_bootstrap(fens, true_values, weight_values, start_epoch=state['epoch_num'], loss=state['error'])
        elif action == 'train_td_endgames':
            print "Resuming endgame TD-Leaf training..."
            weight_values = self.train_td(weight_values, True, game_indices=state['game_indices'], start_idx=state['start_idx'])
        elif action == 'train_td_full':
            print "Resuming full-game TD-Leaf training..."
            weight_values = self.train_td(weight_values, False, game_indices=state['game_indices'], start_idx=state['start_idx'])
        elif action == 'train_selfplay':
            print "Resuming self-play training..."
            weight_values = self.train_selfplay(weight_values, game_indices=state['game_indices'], start_idx=state['start_idx'])
        elif action == 'load_and_resume':
                raise ValueError("Error: It's trying to resume on a resume call - This shouldn't happen.")
        else:
            raise NotImplementedError("Error: %s is not a valid action." % action)

        self.curr_action_idx += 1
        return weight_values

    def out_of_time(self):
        """
        Returns True if training has run out of time. False otherwise
            Output:
                [Boolean]
        """
        return self.training_time is not None and time.time() - self.start_time >= self.training_time

    # ---------- BOOTSTRAP TRAINING METHODS

    def train_bootstrap(self, fens, true_values, weight_values, start_epoch=0, loss=None):
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

        self.nn.set_session(_weight_values=weight_values)

        train_fens = fens[:(-1) * VALIDATION_SIZE]  # fens to train on
        valid_fens = fens[(-1) * VALIDATION_SIZE:]  # fens to check convergence on

        train_values = true_values[:(-1) * VALIDATION_SIZE]
        valid_values = true_values[(-1) * VALIDATION_SIZE:]

        num_boards = len(train_fens)

        if not loss:
            loss = []

        # usr_in = raw_input("This will overwrite your old weights\' pickle, do you still want to proceed (y/n)?: ")
        # if usr_in.lower() != 'y':
        #    return
        print "Training data on %d positions. Will save weights to pickle" % num_boards

        # From my limited understanding x_entropy is not suitable - but if im wrong it could be better
        # Using squared error instead
        cost = tf.reduce_sum(tf.pow(tf.sub(self.nn.pred_value, self.nn.true_value), 2))
        train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cost)
        loss.append(self.evaluate_bootstrap(valid_fens, valid_values))
        for epoch in xrange(start_epoch, NUM_EPOCHS):
            print "Loss for epoch %d: %f" % (epoch+1, loss[-1])
            # Configure data (shuffle fens -> fens to channel -> group batches)
            game_indices = range(num_boards)
            random.shuffle(game_indices)

            # update weights
            save = self.weight_update_bootstrap(train_fens, train_values, game_indices, train_step)

            # save state if timeout
            if save[0]:
                save[1]['epoch_num'] = epoch + 1
                save[1]['error'] = loss
                self.save_state(save[1])
                return

            # evaluate nn for convergence
            loss.append(self.evaluate_bootstrap(valid_fens, valid_values))
            print "%d: %f" % (epoch+1, loss[-1])
            if len(loss) > 2:
                base_loss = loss[0] - loss[1]
                curr_loss = loss[-2] - loss[-1]
                if abs(curr_loss / base_loss) < LOSS_THRESHOLD:
                    print "Training complete: Reached convergence threshold"
                    break
        else:
            print "Training complete: Reached max epoch, no convergence yet"

        # save loss
        pickle.dump(loss, open(self.dir_path + '/../pickles/loss_' + time.strftime("%Y%m%d-%H%M%S") + ".p", 'wb'))
        # plt.plot(range(epoch + 1), error)
        # plt.show()
        
        weight_values = self.nn.close_session()
        return weight_values

    def weight_update_bootstrap(self, fens, true_values_, game_indices, train_step):
        """ Weight update for multiple batches"""

        if len(game_indices) % BATCH_SIZE != 0:
            raise Exception("Error: number of fens provided (%d) is not a multiple of batch_size (%d)" %
                            (len(game_indices), BATCH_SIZE))

        num_batches = int(len(game_indices) / BATCH_SIZE)

        board_num = 0
        boards = np.zeros((BATCH_SIZE, 8, 8, NUM_CHANNELS))
        diagonals = np.zeros((BATCH_SIZE, 10, 8, NUM_CHANNELS))
        true_values = np.zeros(BATCH_SIZE)

        for i in xrange(num_batches):
            # if training time is up, save state
            if self.out_of_time():
                print "Bootstrap Timeout: Saving state and quitting"
                return True, {'game_indices': game_indices[(i * BATCH_SIZE):]}

            # set up batch
            for j in xrange(BATCH_SIZE):
                boards[j] = dh.fen_to_channels(fens[game_indices[board_num]])
                diagonals[j] = dh.get_diagonals(boards[j])
                true_values[j] = true_values_[game_indices[board_num]]
                board_num += 1

            # train batch
            self.nn.sess.run([train_step], feed_dict={self.nn.data: boards, self.nn.data_diags: diagonals,
                                                      self.nn.true_value: true_values})

        return False, {}

    def evaluate_bootstrap(self, fens, true_values):
        """
            Evaluate neural net

            Inputs:
                fens[list of strings]:
                    The fens representing the board states
                true_values[ndarray]:
                    Expected output for each chess board state (between 0 and 1)
        """

        # Create tensors
        pred_value = tf.reshape(self.nn.pred_value, [-1]) # NOWDO: define once with a placeholder
        err = tf.sub(self.nn.true_value, pred_value) # NOWDO: define once with a placeholder
        err_sum = tf.reduce_sum(err) # NOWDO: define once with a placeholder

        # Configure data
        boards = np.zeros((VALIDATION_SIZE, 8, 8, NUM_CHANNELS))
        diagonals = np.zeros((VALIDATION_SIZE, 10, 8, NUM_CHANNELS))
        for i in xrange(VALIDATION_SIZE):
            boards[i] = dh.fen_to_channels(fens[i])
            diagonals[i] = dh.get_diagonals(boards[i])

        # Get loss
        error = self.nn.sess.run([err_sum], feed_dict={
            self.nn.data: boards,
            self.nn.data_diags: diagonals,
            self.nn.true_value: true_values
        })

        return abs(error[0])

    # ---------- TD-LEAF TRAINING METHODS

    # TODO: Handle complete fens format

    def set_td_params(self, num_end=None, num_full=None, randomize=None,
                      pgn_folder=None, end_length=None, full_length=None):
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
                end_depth [Int]
                    Length of endgames.
                full_depth [Int]
                    Maximum length of full games. (-1 for no max)
        """

        if num_end:
            self.td_num_endgame = num_end
        if num_full:
            self.td_num_full = num_full
        if randomize:
            self.td_rand_file = randomize
        if pgn_folder:
            self.td_pgn_folder = pgn_folder
        if end_length:
            self.td_end_length = end_length
        if full_length:
            self.td_full_length = full_length

    def train_td(self, weight_values, endgame, game_indices=None, start_idx=0):
        """
        Trains the neural net using TD-Leaf.
            Inputs:
                endgame [Boolean]
                    If True then only trains on endgames. Otherwise trains on a random subset of full games.
                game_indices [List of Ints]
                    Used when resuming. Provides (shuffled) list of indices for the games to train on.
                start_idx [Int]
                    Index of game_indices where training should be resumed.
        """
        self.nn.set_session(_weight_values=weight_values)

        assert not ((game_indices is None) and start_idx > 0)

        num_games = self.td_num_endgame if endgame else self.td_num_full

        # Only load some files if not random
        if self.td_rand_file:
            pgn_files = [f for f in os.listdir(self.td_pgn_folder) if
                         os.path.isfile(os.path.join(self.td_pgn_folder, f))]
        else:
            pgn_files = [f for f in os.listdir(self.td_pgn_folder)[:num_games] if
                         os.path.isfile(os.path.join(self.td_pgn_folder, f))]

        if game_indices is None:
            game_indices = range(num_games if num_games >= 0 else len(pgn_files))

            # Shuffle if necessary
            if self.td_rand_file:
                random.shuffle(game_indices)

        for i in xrange(start_idx, len(game_indices)):

            # Delete once we've fully transitioned to placeholders and we know that memory isn't going to be overloaded
            if i % 5 == 0 and i != 0:
                # Close and Reopen session every batch to avoid memory overload
                print '-'*30
                print hpy().heap()

            game_idx = game_indices[i]
            print "Training on game %d of %d..." % (i + 1, num_games)
            fens = []

            # Open and use pgn file sequentially or at random
            with open(os.path.join(self.td_pgn_folder, pgn_files[game_idx])) as pgn:
                game = chess.pgn.read_game(pgn)

                if endgame:
                    # Only get endgame fens
                    curr_node = game.end()
                    for _ in range(self.td_end_length):
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

                    # Get random subset if working with a max size
                    if self.td_full_length != -1:
                        game_length = len(fens)
                        sub_start = random.randint(0, max(0, game_length - self.td_full_length))
                        sub_end = min(game_length, sub_start + self.td_full_length)
                        fens = fens[sub_start:sub_end]

                        # TODO: Remove this check later.
                        if (len(fens) != self.td_full_length and game_length >= self.td_full_length) or \
                                (len(fens) != game_length and game_length < self.td_full_length):
                            print "Warning: This shouldn't happen!"

            # Call TD-Leaf
            self.td_leaf(fens)

            # Check if out of time
            if self.out_of_time() and i != (len(game_indices) - 1):
                print "TD-Leaf " + ("endgame" if endgame else "fullgame") + " Timeout: Saving state and quitting."
                save = {'game_indices': game_indices,
                        'start_idx': i + 1}
                self.save_state(save)
                return

        return

    def td_leaf(self, game):
        """
        Trains neural net using TD-Leaf algorithm.
            Inputs:
                Game [List]
                    A game consists of a sequential list of board states. Each board state is a FEN.
        """

        num_boards = len(game)
        game_info = [{'value': None, 'gradient': None} for _ in range(num_boards)]  # Indexed the same as num_boards
        w_update = None

        # Pre-calculate leaf value (J_d(x,w)) of search applied to each board
        # Get new board state from leaf
        # print "Calculating TD-Leaf values for move ",
        for i, board in enumerate(game):
            # print str(i) + "... ",
            value, _, board_fen = self.guerilla.search.run(chess.Board(board))

            # Get values and gradients for white plays next
            if dh.white_is_next(board_fen):
                game_info[i]['value'] = value
                game_info[i]['gradient'] = self.nn.get_all_weights_gradient(board_fen)
            else:
                # value is probability of WHITE winning
                game_info[i]['value'] = 1 - value

                # Get NEGATIVE gradient of a flipped board
                # Explanation:
                #   Flipped board = Black -> White, so now white plays next (as required by NN)
                #   Gradient of flipped board = Gradient of what used to be black
                #   Desired gradient = Gradient of what was originally white = - Gradient of flipped board
                game_info[i]['gradient'] = [-x for x in self.nn.get_all_weights_gradient(dh.flip_board(board_fen))]

        for t in range(num_boards):
            td_val = 0
            for j in range(t, num_boards - 1):
                # Calculate temporal difference
                dt = game_info[j + 1]['value'] - game_info[j]['value']
                # Add to sum
                td_val += math.pow(TD_DISCOUNT, j - t) * dt

            # Use gradient to update sum
            if not w_update:
                w_update = [w * td_val for w in game_info[t]['gradient']]
            else:
                # update each set of weights
                for i in range(len(game_info[t]['gradient'])):
                    w_update[i] += game_info[t]['gradient'][i] * td_val

        # Update neural net weights.
        self.nn.add_all_weights([TD_LRN_RATE * w_update[i] for i in range(len(w_update))])
        # print "Weights updated."

    # ---------- SELF-PLAY TRAINING METHODS

    def set_sp_params(self, num_selfplay=None, max_length=None):
        """
        Set the parameteres for self-play.
            Inputs:
                num_selfplay [Int]
                    Number of games to play against itself and train using TD-Leaf.
                max_length [Int]
                    Maximum number of moves in each self-play. (-1 = No max)
        """

        if num_selfplay:
            self.sp_num = num_selfplay
        if max_length:
            self.sp_length = max_length

    def train_selfplay(self, weight_values, game_indices=None, start_idx=0):
        """
        Trains neural net using TD-Leaf algorithm based on partial games which the neural net plays against itself.
        Self-play is performed from a random board position. The random board position is found by loading from the fens
        file and then applying a random legal move to the board.
        """
        self.nn.set_session(_weight_values=weight_values)

        fens = cgp.load_fens(self.files[0])

        if game_indices is None:
            game_indices = np.random.choice(len(fens), self.sp_num)

        max_len = float("inf") if self.sp_length == -1 else self.sp_length

        for i in xrange(start_idx, len(game_indices)):

            print "Generating self-play game %d of %d..." % (i + 1, self.sp_num)
            # Load random fen
            board = chess.Board(
                fens[game_indices[i]] + " w KQkq - 0 1")  # white plays next, turn counter & castling unimportant here

            # Play random move to increase game variability
            board.push(random.sample(board.legal_moves, 1)[0])

            # Play a game against yourself
            game_fens = [board.fen()]
            for _ in range(max_len):
                # Check if game finished
                if board.is_checkmate():
                    break

                # Play move
                board.push(self.guerilla.get_move(board))

                # Store fen
                game_fens.append(board.fen())

            # Send game for TD-leaf training
            print "Training on game %d of %d..." % (i + 1, self.sp_num)
            self.td_leaf(game_fens)

            # Check if out of time
            if self.out_of_time() and i != (len(game_indices) - 1):
                print "TD-Leaf self-play Timeout: Saving state and quitting."
                save = {'game_indices': game_indices,
                        'start_idx': i + 1}
                self.save_state(save)
                return

        return self.nn.close_session()


def main():
    g = guerilla.Guerilla('Harambe', 'w')#, _load_file='weights_train_bootstrap_20160927-025555.p')
    g.search.max_depth = 1
    t = Teacher(g)
    t.set_td_params(num_end=40, num_full=20, randomize=False, end_length=5, full_length=12)
    t.set_sp_params(num_selfplay=1000, max_length=12)
    t.run(['train_td_endgames'], training_time=None, fens_filename="fens_1000.p", stockfish_filename="true_values_1000.p")
    # t.run(['load_and_resume'], training_time=72000)


if __name__ == '__main__':
    main()