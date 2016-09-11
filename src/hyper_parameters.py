"""
Hyper parameters, constants and definitions.
"""

# TODO S: Add definitions and explanations in comments.

NUM_FEAT = 10  # TODO S: Rename to "filters" or something else. Currently confusing.
NUM_EPOCHS = 2
BATCH_SIZE = 5
NUM_HIDDEN = 1024
LEARNING_RATE = 0.001

NUM_CHANNELS = 6 * 2

# TODO: I don't think this should be here.
piece_indices = {
    'p': 0,
    'r': 1,
    'n': 2,
    'b': 3,
    'q': 4,
    'k': 5,
}
