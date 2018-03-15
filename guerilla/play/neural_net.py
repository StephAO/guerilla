import pickle

import numpy as np
import tensorflow as tf
import ruamel.yaml
from pkg_resources import resource_filename

import guerilla.data_handler as dh


class NeuralNet:
    training_modes = ['adagrad', 'adadelta', 'gradient_descent']

#######################
### INITIALIZATIONS ###
#######################

    def __init__(self, load_file=None, hp_load_file=None, seed=None, verbose=True, hp=None):
        """
            Initializes neural net. Generates session, placeholders, variables,
            and structure.
            Input:
                load_file [String]:
                    The filename from which the neural network weights should be loaded.
                    If 'None' then the weights are randomly initialized.
                hp_load_file [String]:
                    The filename from which hyper parameters should be laoded from.
                    If 'None' then keyword arguments will be used (if not
                    provided, then default).
                seed [Int]
                    Value for the graph-level seed. If 'None', seed is not set. Default is 'None'.
                verbose [Bool]:
                    Enables Verbose mode.
                hp [Dict]:
                    Hyper parameters in keyword format. Keyword must match hyper
                    parameter name. See self._set_hyper_params for valid hyper
                    parameters. Hyper parameters defined in hp will overwrite
                    params loaded from a file.
        """
        self.load_file = load_file
        self.verbose = verbose
        self.hp = {}

        # set random seed
        if seed is not None:
            tf.set_random_seed(seed)

        if hp_load_file is None:
            hp_load_file = 'default.yaml'
        self._set_hyper_params_from_file(hp_load_file)
        if hp is None:
            hp = {}
        self._set_hyper_params(hp)

        # Always list different input structures in increasing order of size
        # All input sizes must be tuples. If it's a single value, use (x,)
        # Currently, only single one of the inputs can be used for convolution
        # The nn_input_type must have a function in data_handler named:
        # fen_to_<nn_input_type>
        self.total_input_size = 0
        if self.hp['NN_INPUT_TYPE'] == 'movemap':
            self.weight_stddev = 0.01
            self.input_sizes = [((dh.STATE_DATA_SIZE,), 0.125),
                                ((dh.BOARD_LENGTH, dh.BOARD_LENGTH, dh.MOVEMAP_TILE_SIZE), 0.875)]
            # Used for convolution
            self.size_per_tile = dh.MOVEMAP_TILE_SIZE
        elif self.hp['NN_INPUT_TYPE'] == 'giraffe':
            self.weight_stddev = 0.001
            self.input_sizes = [((dh.STATE_DATA_SIZE,), 0.125),
                                ((dh.BOARD_DATA_SIZE,), 0.5), ((dh.PIECE_DATA_SIZE,), 0.375)]
            self.hp['USE_CONV'] = False
        elif self.hp['NN_INPUT_TYPE'] == 'bitmap':
            self.weight_stddev = 0.01
            self.input_sizes = [((dh.BOARD_LENGTH, dh.BOARD_LENGTH, dh.BITMAP_TILE_SIZE), 1.0)]
            # Used for convolution
            self.size_per_tile = dh.BITMAP_TILE_SIZE
        else:
            raise NotImplementedError("Neural Net input type %s is not implemented" % (self.hp['NN_INPUT_TYPE']))

        # Dropout keep probability placeholder -> By default does not dropout when session is run
        self.kp_default = tf.constant(1.0, dtype=tf.float32)
        self.keep_prob = tf.placeholder_with_default(self.kp_default, self.kp_default.get_shape())

        # declare layer variables
        self.sess = None

        self.all_weights = []
        self.all_biases = []
        self.all_weights_biases = []

        # all weights + biases
        # Currently the order is necessary for assignment operators
        self.bias_pl = []
        self.weight_pl = []
        self.all_placeholders = []

        # declare output variable
        self.pred_value = None

        # tf session and variables
        self.sess = None

        self.conv_layer_size = 64 # 90

        # input placeholders
        self.true_value = tf.placeholder(tf.float32, shape=[None])
        self.input_data_placeholders = []
        for input_size in self.input_sizes:
            self.total_input_size += float(np.prod(input_size[0]))
            _shape = [None] + list(input_size[0])

            self.input_data_placeholders.append(tf.placeholder(
                tf.float32, shape=_shape))

        # create neural net graph
        self.model()

        # gradient op and placeholder (must be defined after self.pred_value is defined)
        self.all_weights_biases.extend(self.all_weights)
        self.all_weights_biases.extend(self.all_biases)

        self.all_placeholders.extend(self.weight_pl)
        self.all_placeholders.extend(self.bias_pl)

        # create assignment operators
        self.all_assignments = []
        for i in xrange(len(self.all_weights_biases)):
            self.all_assignments.append(self.all_weights_biases[i].assign(
                self.all_placeholders[i]))

        if len(self.all_weights_biases) != len(self.all_placeholders):
            raise ValueError("There are an unequal number of weights (%d) and"
                             " placeholders for those weights" %
                             len(self.all_weights_biases), len(self.all_placeholders))

        self.grad_all_op = tf.gradients(self.pred_value, self.all_weights_biases)

        self.global_step = tf.Variable(0) # used for learning rate decay

        # Blank training variables. Call 'init_training' to initialize them
        self.train_optimizer = None

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
            self.sess.run(tf.variables_initializer(set(tf.global_variables()) - set(self.all_weights_biases)))
        else:
            if self.verbose:
                print "Initializing variables from a normal distribution."
            self.sess.run(tf.global_variables_initializer())

    def init_training(self, training_mode, learning_rate, reg_const, loss_fn, batch_size, decay_rate=None):
        """
        Initializes the training optimizer, loss function and training step.
        Input:
            training_mode [String]
                Training mode to use. See NeuralNet.training_modes for options.
            learning_rate [Float]
                Learning rate to use in training.
            reg_const [Float]
                Regularization constant. To not use regularization simply input 0.
            loss_fn [Tensor]
                Loss function to use.
            decay_rate [String]
                Decay rate. Input is only necessary when training mode is 'adadelta'.
        """

        if training_mode not in NeuralNet.training_modes:
            raise ValueError("Invalid training mode input!" \
                             " Please refer to NeuralNet.training_modes " \
                             "for valid inputs.")

        # Regularization Term
        regularization = tf.add_n([tf.nn.l2_loss(w) for w in self.all_weights]) * reg_const

        base_learning_rate = learning_rate
        # Exponentionally decaying learning rate
        learning_rate = tf.train.exponential_decay(learning_rate,  # Base learning rate.
                                                   self.global_step * batch_size,  # Current index into the dataset.
                                                   self.hp["LEARNING_RATE_DECAY_STEP"], # Decay step.
                                                   self.hp["LEARNING_RATE_DECAY_RATE"],  # Decay rate.
                                                   staircase=True)
        learning_rate = tf.maximum(learning_rate, 0.01 * base_learning_rate) # clip learning rate

        # Set tensorflow training method for bootstrap training
        if training_mode == 'adagrad':
            self.train_optimizer = tf.train.AdagradOptimizer(learning_rate)
        elif training_mode == 'adadelta':
            if decay_rate is None:
                raise ValueError("When the training mode is 'adadelta' the decay rate must be specified!")
            self.train_optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate, rho=decay_rate)
        elif training_mode == 'gradient_descent':
            self.train_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = self.train_optimizer.minimize(loss_fn + regularization, global_step=self.global_step)
        self.train_saver = tf.train.Saver(
            var_list=self.get_training_vars())  # TODO: Combine var saving with "in_training" weight saving

        # initialize training variables if necessary
        train_vars = self.get_training_vars()
        if train_vars is not None:
            self.sess.run(tf.variables_initializer(train_vars.values()))

        return train_step, self.global_step

###############
### CONTROL ###
###############

    def start_session(self):
        """ Starts tensorflow session """

        assert self.sess is None

        self.sess = tf.Session()
        # self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
        if self.verbose:
            print "Tensorflow session opened."

    def close_session(self):
        """ Closes tensorflow session."""
        assert self.sess is not None

        self.sess.close()
        if self.verbose:
            print "Tensorflow session closed."

    def reset_graph(self):
        """ Resets the defaults graph."""
        tf.reset_default_graph()
        if self.verbose:
            print "Default graph reset."

######################
### VARIABLE TYPES ###
###################### 
    def weight_variable(self, name, shape, use_xavier=True, wd=None):
        """
        Create an initialized weight Variable with weight decay.
        Args:
            shape[list of ints]: shape of variable
            use_xavier[bool]: whether to use xavier initializer
        Returns:
            Variable Tensor
        """
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=self.weight_stddev)
        weight = tf.get_variable(name, shape, initializer=initializer)
        self.all_weights.append(weight)
        self.weight_pl.append(tf.placeholder(tf.float32, shape=shape))
        return weight

    def bias_variable(self, name, shape):
        """
        Create an initialized bias variable.
        Inputs:
            shape [list of ints]: shape of variable
        Returns:
            Variable Tensor
        """
        initializer = tf.constant_initializer(self.weight_stddev)
        bias = tf.get_variable(name, shape, initializer=initializer)
        self.all_biases.append(bias)
        self.bias_pl.append(tf.placeholder(tf.float32, shape=shape))
        return bias

######################
### NN LAYER TYPES ###
###################### 
# (see https://github.com/okraus/DeepLoc/blob/master/nn_layers.py) on how to extend them
    def fc_layer(self, input_tensor, input_dim, output_dim, layer_name,
                 activation_fn=tf.nn.relu, is_training=True, batch_norm=False,
                 batch_norm_decay=None):
        """
        Reusable code for making a simple neural net layer.
        It does a matrix multiply, bias add, and then uses relu to nonlinearize.
        It also sets up name scoping so that the resultant graph is easy to read,
        and adds a number of summary ops.
        """
        # Adding a name scope ensures logical grouping of the layers in the graph.
        with tf.name_scope(layer_name):
            weights = self.weight_variable(layer_name + "_weights", [input_dim, output_dim])
            biases = self.bias_variable(layer_name + "_biases", [output_dim])
            output = tf.matmul(input_tensor, weights) + biases
            if batch_norm:
                    output = self.batch_norm_fc(output, is_training=is_training,
                                           bn_decay=batch_norm_decay,
                                           scope=layer_name+'_batch_norm')
            if activation_fn is not None:
                    output = activation_fn(output, name='activation')
            return output


    def conv2d(self, input_tensor, num_in_feat_maps, num_out_feat_maps, kernel_size,
               layer_name, stride=[1, 1], padding='SAME', use_xavier=False,
               stddev=1e-3, activation_fn=tf.nn.relu, batch_norm=False,
               batch_norm_decay=None, is_training=None):
        """
        2D convolution with non-linear operation.
        Args:
            input_tensor: 4-D tensor variable BxHxWxC
            num_in_feat_maps: int
            num_out_feat_maps: int
            kernel_size: a list of 2 ints
            layer_name: string used to scope variables in layer
            stride: a list of 2 ints
            padding: 'SAME' or 'VALID'
            use_xavier: bool, use xavier_initializer if true
            stddev: float, stddev for truncated_normal init
            activation_fn: function
            batch_norm: bool, whether to use batch norm
            batch_norm_decay: float or float tensor variable in [0,1]
            is_training: bool Tensor variable
        Returns:
            Variable tensor
        """
        with tf.variable_scope(layer_name) as sc:
            kernel_h, kernel_w = kernel_size
            kernel_shape = [kernel_h, kernel_w, num_in_feat_maps, num_out_feat_maps]
            weights = self.weight_variable(layer_name + "_weights", shape=kernel_shape)
            stride_h, stride_w = stride
            output = tf.nn.conv2d(input_tensor, weights,
                                   [1, stride_h, stride_w, 1],
                                   padding=padding)
            biases = self.bias_variable(layer_name + "_biases", [num_out_feat_maps])
            output = tf.nn.bias_add(output, biases)

            if batch_norm:
                output = self.batch_norm_conv2d(output, is_training,
                                            bn_decay=batch_norm_decay, scope='bn')
            if activation_fn is not None:
                output = activation_fn(output)

            return output

    def max_pool2d(self, input_tensor, kernel_size, layer_name, stride=[2, 2],
                   padding='VALID'):
        """
        2D max pooling.
        Args:
            input_tensor: 4-D tensor BxHxWxC
            kernel_size: a list of 2 ints
            layer_name: string to scope variables
            stride: a list of 2 ints
            padding: string, either 'VALID' or 'SAME'
        Returns:
            Variable tensor
        """
        with tf.variable_scope(layer_name) as sc:
            kernel_h, kernel_w = kernel_size
            stride_h, stride_w = stride
            output = tf.nn.max_pool(input_tensor,
                                     ksize=[1, kernel_h, kernel_w, 1],
                                     strides=[1, stride_h, stride_w, 1],
                                     padding=padding)
            return output


    def batch_norm_template(self, inputs, is_training, scope, moments_dims, bn_decay):
        """ 
        Batch normalization on convolutional maps and beyond...
        Ref.: http://stackoverflow.com/questions/33949786/how-could-i-use-batch-normalization-in-tensorflow
        Args:
            inputs:        Tensor, k-D input ... x C could be BC or BHWC or BDHWC
            is_training:   boolean tf.Varialbe, true indicates training phase
            scope:         string, variable scope
            moments_dims:  a list of ints, indicating dimensions for moments calculation
            bn_decay:      float or float tensor variable, controling moving average weight
        Return:
            normed:        batch-normalized maps
        """

        # TODO compare with and without batch norm
        # TODO Will these have to be considered by TD?
        with tf.variable_scope(scope) as sc:
            num_channels = inputs.get_shape()[-1].value
            beta = tf.Variable(tf.constant(0.0, shape=[num_channels]),
                               name='beta', trainable=True)
            gamma = tf.Variable(tf.constant(1.0, shape=[num_channels]),
                                name='gamma', trainable=True)
            batch_mean, batch_var = tf.nn.moments(inputs, moments_dims,
                                                  name='moments')
            decay = bn_decay if bn_decay is not None else 0.9
            ema = tf.train.ExponentialMovingAverage(decay=decay)
            # Operator that maintains moving averages of variables.
            ema_apply_op = tf.cond(is_training,
                                   lambda: ema.apply([batch_mean, batch_var]),
                                   lambda: tf.no_op())

            # Update moving average and return current batch's avg and var.
            def mean_var_with_update():
                with tf.control_dependencies([ema_apply_op]):
                    return tf.identity(batch_mean), tf.identity(batch_var)

            # ema.average returns the Variable holding the average of var.
            mean, var = tf.cond(is_training,
                                mean_var_with_update,
                                lambda: (
                                ema.average(batch_mean), ema.average(batch_var)))
            normed = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, 1e-3)
        return normed


    def batch_norm_fc(self, inputs, is_training, bn_decay, scope):
        """ 
        Batch normalization on FC data.
        Args:
            inputs:      Tensor, 2D BxC input
            is_training: boolean tf.Varialbe, true indicates training phase
            bn_decay:    float or float tensor variable, controling moving average weight
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        return self.batch_norm_template(inputs, is_training, scope, [0, ], bn_decay)

    def batch_norm_conv2d(self, inputs, is_training, bn_decay, scope):
        """ 
        Batch normalization on 2D convolutional maps.
        Args:
            inputs:      Tensor, 4D BHWC input maps
            is_training: boolean tf.Varialbe, true indicates training phase
            bn_decay:    float or float tensor variable, controling moving average weight
            scope:       string, variable scope
        Return:
            normed:      batch-normalized maps
        """
        return self.batch_norm_template(inputs, is_training, scope, [0, 1, 2], bn_decay)

#############
### MODEL ###
#############
    def mae_loss(self, batch_size):
        """
        Returns a MAE loss Tensor
        Input:
            batch_size [Int]
                Batch size.
        Output:
            mse [Tensor]
                Mean absolute error loss.
        """

        err = tf.reduce_sum(tf.abs(tf.subtract(
            tf.reshape(self.pred_value, shape=tf.shape(self.true_value)), self.true_value)))

        return tf.div(err, batch_size)

    def mse_loss(self, batch_size):
        """
        Returns a MSE loss Tensor
        Input:
            batch_size [Int]
                Batch size.
        Output:
            mse [Tensor]
                Mean squared error loss.
        """

        err = tf.reduce_sum(tf.pow(tf.subtract(
            tf.reshape(self.pred_value, shape=tf.shape(self.true_value)), self.true_value), 2))

        return tf.div(err, batch_size)

    def model(self):
        """
            Structure of neural net.
            Sets member variable 'pred_value' to the tensor representing the
            output of neural net.
        """
        batch_size = tf.shape(self.input_data_placeholders[0])[0]

        mid_output = []
        num_mid_nodes = 0
        for i, input_size in enumerate(self.input_sizes):
            # convolve layer if you can and desired
            if input_size[0][0:2] == (8, 8) and self.hp['USE_CONV']:

                conv1 = self.conv2d(self.input_data_placeholders[i], self.size_per_tile, self.hp['NUM_FEAT'], [5, 5],
                                    'conv1_' + str(i))
                conv2 = self.conv2d(conv1, self.hp['NUM_FEAT'], 2 * self.hp['NUM_FEAT'], [3, 3],
                                    'conv2_' + str(i))
                conv2 = self.conv2d(conv2, 2 * self.hp['NUM_FEAT'], 4 * self.hp['NUM_FEAT'], [3, 3],
                                    'conv3_' + str(i))

                mid_output.append(tf.reshape(conv2, [batch_size, 64 * 4 * self.hp['NUM_FEAT']]))
                num_mid_nodes += 64 * 4 * self.hp['NUM_FEAT']

            else:
                input_shape = [batch_size, np.prod(input_size[0])]
                input_tensor = tf.reshape(self.input_data_placeholders[i], input_shape)
                l1 = self.fc_layer(input_tensor, np.prod(input_size[0]), int(self.hp['NUM_HIDDEN'] * input_size[1]),
                                   "layer1_" + str(i))

                mid_output.append(tf.reshape(l1, [batch_size, int(self.hp['NUM_HIDDEN'] * input_size[1])]))
                num_mid_nodes += self.hp['NUM_HIDDEN'] * input_size[1]


        # output of first layer
        mid_output = tf.concat(values=mid_output, axis=1)
        # output of fully connected layers
        fc1 = self.fc_layer(mid_output, num_mid_nodes, self.hp['NUM_HIDDEN'], "fc1")
        fc2 = self.fc_layer(fc1, self.hp['NUM_HIDDEN'], self.hp['NUM_HIDDEN'] / 2 , "fc2")

        # final_output
        self.pred_value = self.fc_layer(fc2, self.hp['NUM_HIDDEN'] / 2, 1, "predicted_value", activation_fn=None)

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

        output = self.pred_value.eval(feed_dict=self.board_to_feed(fen), session=self.sess)[0][0]

        if np.isnan(output):
            raise RuntimeError("Neural network output NaN! Most likely due to bad training parameters.")
        if np.isinf(output):
            raise RuntimeError("Neural network output %s infinity! Most likely due to bad training parameters." %
                               ('positive' if output > 0 else 'negative'))

        return output

#######################
### GETTERS/SETTERS ###
#######################

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

    def get_all_weights(self):
        """
        Get all weights and biases.
            Output:
                weights [List]
                    Weights & biases.
        """

        return self.get_weights(self.all_weights_biases)

    def set_all_weights(self, weight_vals):
        """
        NOTE: currently only supports updating all weights, must be in the same order.
        Updates the neural net weights based on the input.
            Input:
                weight_vals [List]
                    List of values with which to update weights. Must be in desired order.
        """
        if len(weight_vals) != len(self.all_weights_biases):
            raise ValueError("Error: There are a different number of " \
                             "weights(%d) and weight values(%d)" % \
                             (len(self.all_weights_biases), len(weight_vals)))

        # match value to placeholders
        placeholder_dict = dict()
        for i, placeholder in enumerate(self.all_placeholders):
            placeholder_dict[placeholder] = weight_vals[i]

        # Run assignment/update
        self.sess.run(self.all_assignments, feed_dict=placeholder_dict)

    def add_to_all_weights(self, weight_vals):
        """
        Increments all the weight values by the input amount.
            Input:
                weight_vals [List]
                    List of values with which to update weights. Must be in desired order.
        """

        old_weights = self.get_all_weights()
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

    def get_weight_values(self):
        """
            Returns values of weights as a dictionary
        """
        weight_values = dict()

        # Get all weight values
        weight_values['weights'] = [None] * len(self.all_weights)
        result = self.get_weights(self.all_weights)
        for i in xrange(len(result)):
            weight_values['weights'][i] = result[i]

        # Get all bias values
        weight_values['biases'] = [None] * len(self.all_biases)
        result = self.get_weights(self.all_biases)
        for i in xrange(len(result)):
            weight_values['biases'][i] = result[i]

        return weight_values

    def get_training_vars(self):
        """
        Returns the training variables associated with the current training mode.
        Returns None if there are no associated variables.
        Output:
            var_dict [Dict] or [None]:
                Dictionary of variables.
        """
        var_dict = dict()
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        slot_names = self.train_optimizer.get_slot_names()  # Result should just be accumulator
        for name in slot_names:
            for var in train_vars:
                val = self.train_optimizer.get_slot(var, name)
                if val:
                    var_dict[val.name] = val

        if var_dict == {}:
            return None

        return var_dict

#################
### SAVE/LOAD ###
#################

    def save_training_vars(self, path):
        """
        Saves the training variables associated with the current training mode to a file in path.
        Returns the file name.
        Input:
            path [String]:
                Path specifying where the variables should be saved.
        Ouput:
            filename [String]:
                Filename specifying where the training variables were saved.
        """
        filename = None
        if isinstance(self.train_optimizer, tf.train.GradientDescentOptimizer):
            return None
        else:
            filename = self.train_saver.save(self.sess, path)

        if self.verbose:
            print "Saved training vars to %s" % filename
        return filename

    def load_training_vars(self, filename):
        """
        Loads the training variable associated with the current training mode.
        Input:
            filename [String]:
                Filename where training variables are stored.
        """
        if isinstance(self.train_optimizer, tf.train.GradientDescentOptimizer):
            return None
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

        weight_values.extend(values_dict['weights'])
        weight_values.extend(values_dict['biases'])

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

###############
### HELPERS ###
###############

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
        feed_dict = {}
        input_data = dh.fen_to_nn_input(fen, self.hp['NN_INPUT_TYPE'])
        if len(input_data) != len(self.input_data_placeholders):
            raise ValueError("The length of input data (%d) does not equal the " \
                             "length of input data place holders(%d)" % \
                             (len(input_data), len(self.input_data_placeholders)))

        for i in xrange(len(input_data)):
            feed_dict[self.input_data_placeholders[i]] = np.array([input_data[i]])

        return feed_dict

    def _set_hyper_params_from_file(self, file):
        """
            Updates hyper parameters from a yaml file.
            Will only affect hyper parameters that are provided. Unspecified
            hyper parameters will not change.
            WARNING: This can only be called at the start of init, since
                     that is where the shape of the neural net is defined.

            Inputs:
                file[String]:
                    filename to use. File must be in data/hyper_params/teacher/
        """
        relative_filepath = 'data/hyper_params/neural_net/' + file
        filepath = resource_filename('guerilla', relative_filepath)
        with open(filepath, 'r') as yaml_file:
            self.hp.update(ruamel.yaml.safe_load(yaml_file))

    def _set_hyper_params(self, hyper_parameters):
        """
            Updates hyper parameters from arguments.
            Will only affect hyper parameters that are provided. Unspecified
            hyper parameters will not change.
            WARNING: This can only be called at the start of init, since
                     that is where the shape of the neural net is defined.

            Hyper parameters that are used:
                "NUM_FEAT" - Number of times the output nodes of the convolution
                             are repeated
                "NN_INPUT_TYPE" - How the state of the chess board is presented
                                  to the neural net. options are:
                                  1. "bitmap"
                                  2. "giraffe"
                "MIN_NUM_NODES" - Minimum number of nodes for a given input type
                "NUM_HIDDEN" - Number of hidden nodes used in FC layers
                "NUM_FC" - Number of fully connected (FC) layers
                           Excludes any convolutional layers
                "USE_CONV" - Use convolution for bitmap representation

            Inputs:
                hyperparmeters[dict]:
                    hyperparameters to update with
        """
        self.hp.update(hyper_parameters)


if __name__ == 'main':
    pass