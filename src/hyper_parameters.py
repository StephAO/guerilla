"""
Hyper parameters, constants and definitions.
"""

# TODO S: Add definitions and explanations in comments.

NUM_FEAT = 10  # TODO S: Rename to "filters" or something else. Currently confusing.
NUM_EPOCHS = 10
BATCH_SIZE = 50
NUM_HIDDEN = 1024
LEARNING_RATE = 0.0000001
VALIDATION_SIZE = 1000
LOSS_THRESHOLD = 0.0001

NUM_CHANNELS = 6 * 2

# Hyper parameters for TD-Leaf Training Algorithm
TD_LRN_RATE = 0.5  # Learning rate
TD_DISCOUNT = 0.7  # Discount rate

# TODO: I don't think this should be here.
piece_indices = {
    'p': 0,
    'r': 1,
    'n': 2,
    'b': 3,
    'q': 4,
    'k': 5,
}
