"""
Hyper parameters, constants and definitions.
"""

# TODO S: Add definitions and explanations in comments.
# NOTE: VALIDATION SIZE must be a multiple of BATCH SIZE
# NOTE: VALIDATION SIZE + BATCH SIZE < Number of fens provided

hp = {}

hp['NUM_FEAT'] = 10  # TODO S: Rename to "filters" or something else. Currently confusing.
hp['NUM_EPOCHS'] = 1000 #50
hp['BATCH_SIZE'] = 500 #5000
hp['NUM_HIDDEN'] = 1024
hp['NUM_FC_LAYERS'] = 3 # This is in addition to (i.e. excluding) convolutional layers 
hp['LEARNING_RATE'] = 0.0001 #0.0005
hp['VALIDATION_SIZE'] = 5000 #50k
hp['TRAIN_CHECK_SIZE'] = 5000 #50k
hp['LOSS_THRESHOLD'] = 0.000001
hp['DECAY_RATE'] = 0.95 # Only used if the training mode is AdaDelta

hp['NUM_CHANNELS'] = 6 * 2

# Hyper parameters for TD-Leaf Training Algorithm
hp['TD_LRN_RATE'] = 0.00001  # Learning rate
hp['TD_DISCOUNT'] = 0.7  # Discount rate

# Neural net input type. Options are:
# 1. 'bitmap'
# 2. 'position_description'
hp['NN_INPUT_TYPE'] = 'position_description'