import time
import chess
import numpy as np
from guerilla.hyper_parameters import *

piece_indices = {
    'q': 0,
    'r': 1,
    'b': 2,
    'n': 3,
    'p': 4,
    'k': 5,
}

piece_values = {
    'q': 9,
    'r': 5,
    'b': 3,
    'n': 3,
    'p': 1,
    'k': 1000
}

S_IDX_PIECE_LIST = 15
S_IDX_ATKDEF_MAP = 223
GF_FULL_SIZE = 351

def flip_board(fen):
    """ switch colors of pieces
        input:
            fen:
                fen string (only board state)
        output:
            new_fen:
                fen string with colors switched
    """
    board_fen, turn, castling, en_passant, half_clock, full_clock = fen.split()

    # board fen
    new_board_fen = ''
    for char in board_fen:
        if char.isupper():
            new_board_fen += char.lower()
        elif char.islower():
            new_board_fen += char.upper()
        else:
            new_board_fen += char
    new_fen_list = new_board_fen.split('/')
    new_board_fen = '/'.join(new_fen_list[::-1])

    # turn
    turn = 'w' if turn == 'b' else 'b'

    # castling
    new_white_castling = ''
    new_black_castling = ''
    for char in castling:
        if char.islower():
            new_white_castling += char.upper()
        else:
            new_black_castling += char.lower()
    new_castling = new_white_castling + new_black_castling

    # en_passant
    new_en_passant = ''
    if en_passant != '-':
        new_en_passant += en_passant[0]
        new_en_passant += str(9 - int(en_passant[1]))
    else:
        new_en_passant = '-'

    return ' '.join((new_board_fen, turn, new_castling, new_en_passant, half_clock, full_clock))

def fen_to_nn_input(fen):
    if hp['NN_INPUT_TYPE'] == 'bitmap':
        return fen_to_bitmap(fen)
    elif hp['NN_INPUT_TYPE'] == 'giraffe':
        return fen_to_giraffe(fen)
    else:
        raise NotImplementedError("Error: Unsupported Neural Net input type.")

# TODO: deal with en passant and castling
def fen_to_bitmap(fen):
    """
        Converts a fen string to bitmap channels for neural net.
        Always assumes that it's white's turn

        Inputs:
            fen[string]:
                fen string describing current state. 
                Currently only using board state

        Output:
            Channels[ndarray]:

                Consists of 12 8x8 channels (12 8x8 chess boards)
                12 Channels: 6 for each you and your opponents piece types
                Types in order are: Pawns, Rooks, Knights, Bishops, Queens, King
                First 6 channels are white pieces, last 6 are black.
                Index [0,0] corresponds to rank 1 and file a; [8,8] to rank 8 and file h.
    """

    # fen = fen.split(' ')
    # board_str = fen[0]
    # turn = fen[1]
    # castling = fen[2]
    # en_passant = fen[3]

    channels = np.zeros((8, 8, hp['NUM_CHANNELS']))

    c_file = 0
    c_rank = 7
    for char in fen:
        if char == ' ':
            break
        if char == '/':
            c_file = 0
            c_rank -= 1
            continue
        elif char.isdigit():
            c_file += int(char)
            continue
        else:
            white = char.isupper()
            char = char.lower()
            if white:
                channels[c_rank, c_file, piece_indices[char]] = 1
            else:
                channels[c_rank, c_file, piece_indices[char] + 6] = 1

        c_file += 1
        if c_rank == 0 and c_file == 8:
            break
    return channels

def in_bounds(rank_idx, file_idx):
    # Returns True if within the chess board boundary, False otherwise
    # NOTE: Assumes 0 indexed
    return (0 <= rank_idx < 8) and (0 <= file_idx < 8)

def check_range_of_motion(c_rank, c_file, piece, occupied_bitmap, piece_value, \
                          board_to_piece_index, slide_index, map_base_index, gf):
    # TODO: Add description
    # rank (left, right), file (down, up), '/' diag (down-left, up-right) , '\' diag (up-left, down-right)
    still_sliding = [True] * 8 if piece == 'q' else [True] * 4

    crosswise = [
        (lambda x: np.array([0, -x - 1]), 0), # left
        (lambda x: np.array([0, +x + 1]), 1), # right
        (lambda x: np.array([-x - 1, 0]), 2), # down
        (lambda x: np.array([+x + 1, 0]), 3)  # up
    ]

    diagonals = [
        (lambda x: np.array([-x - 1, -x - 1]), 4), # down left
        (lambda x: np.array([+x + 1, +x + 1]), 5), # up right
        (lambda x: np.array([+x + 1, -x - 1]), 6), # up left
        (lambda x: np.array([-x - 1, +x + 1]), 7)  # down right
    ]

    directions = []
    if (piece == 'q' or piece == 'r'):
        directions += crosswise
    if (piece == 'q' or piece == 'b'):
        directions += diagonals

    pos = np.array([c_rank, c_file])
    
    for offset in xrange(0, 8):
        # check horizontal and vertical sliding
        for i, direction in enumerate(directions):
            # tile to check
            r, f = pos + direction[0](offset)
            # If end of slide (piece in the way or out of bounds)
            if still_sliding[i] \
                and (not in_bounds(r, f) or occupied_bitmap[r][f] != 0):
                still_sliding[i] = False
                gf[slide_index + i] = offset
                # If stopped by a piece, set defender/attacker map
                if in_bounds(r, f):
                    defender = (occupied_bitmap[c_rank][c_file] \
                             == occupied_bitmap[r][f])
                    map_index = map_base_index + (r * 8 + f) * 2
                    map_index += 0 if defender else 1
                    gf[map_index] = min(piece_value, gf[map_index])
                    gf[board_to_piece_index[(r, f)] + (3 if defender else 4)] = \
                        gf[map_index]

            if not any(still_sliding):
                break

def set_att_def_map(c_rank, c_file, piece, occupied_bitmap, piece_value, \
                    board_to_piece_index, map_base_index, gf):

    white = (occupied_bitmap[c_rank][c_file] == 1)
    moves_dict = {'n': [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)],
                  'p': [(1, 1), (1, -1)] if white else [(-1, 1), (-1, -1)],
                  'k': [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]}

    if piece not in moves_dict:
        raise ValueError("Invalid piece input! Piece type: %s" % (piece))

    for i, j in moves_dict[piece]:
        r, f = c_rank + i, c_file + j
        if in_bounds(r, f) and occupied_bitmap[r][f] != 0:
            defender = (occupied_bitmap[c_rank][c_file] \
                     == occupied_bitmap[r][f])  # True if defending
            map_index = map_base_index + (r * 8 + f) * 2
            map_index += 0 if defender else 1
            gf[map_index] = min(piece_value, gf[map_index])
            gf[board_to_piece_index[(r, f)] + (3 if defender else 4)] = \
                gf[map_index]

def fen_to_giraffe(fen):
    # TODO docstring
    piece_desc_index = {
        'q': 0,
        'r': 1,
        'b': 3,
        'n': 5,
        'p': 7,
        'k': 15
    }

    piece_index_to_slide_index = {
        15: 175,
        20: 183,
        25: 187,
        30: 191,
        35: 195,
        95: 199,
        100: 207,
        105: 211,
        110: 215,
        115: 219,
    }

    S_IDX_PIECES_NUM = 5
    S_IDX_SLIDE_LIST = 175
    NUM_SLOTS_PER_PIECE = 5 # For piece list
    NUM_SLIDE_PIECES_PER_SIDE = 5 # queen + 2 rooks + 2 bishops
    NUM_PIECES_PER_SIDE = 16 # PER SIDE
    BOARD_LENGTH = 8
    BOARD_SIZE = 64

    # Side to Move (0)
    # Castling Rights (1-4)
    # Material Configuration (5-14)
    # Piece list (15-174)
    # Sliding list (175-222)
    # Def/Atk map (223-250)
    gf = [0] * S_IDX_PIECE_LIST # TODO: What does gf stand for? # ANSWER (delete once you've seen it) gf is short for giraffe
    for i in xrange(NUM_PIECES_PER_SIDE * 2):
        gf += [0, 0, 0, 999999, 999999]
    gf += [0] * (S_IDX_ATKDEF_MAP - S_IDX_SLIDE_LIST)
    gf += [999999] * (BOARD_SIZE * 2) # Attack and defend maps

    fen = fen.split(' ')
    board_str = fen[0]
    turn = fen[1]
    castling = fen[2]
    # en_passant = fen[3]
    
    # Used for sliding and attack/defense maps
    # +1 if white piece, -1 if black piece, 0 o/w
    occupied_bitmap = [[0] * BOARD_LENGTH for _ in range(BOARD_LENGTH)] 
    board_to_piece_index = {} # Key: Coordinate, Value: Piece location in gf
    board_to_piece_type = {} # Key: Coordinate, Value: Piece type

    # Slide to move
    gf[0] = 1 if (turn == 'w') else 0

    # Castling rights
    gf[1] = 1 if ('Q' in castling) else 0
    gf[2] = 1 if ('K' in castling) else 0
    gf[3] = 1 if ('q' in castling) else 0
    gf[4] = 1 if ('k' in castling) else 0

    # Iterate through ranks starting from rank 1
    ranks = board_str.split('/')
    ranks.reverse()
    for c_rank, rank in enumerate(ranks):
        c_file = 0 # File count
        for char in rank:
            if char.isdigit():
                # Increment file count when empty squares are encountered
                c_file += int(char) - 1
            else:    
                white = char.isupper()
                char = char.lower()

                # Update material configuration
                if char != 'k':
                    gf[S_IDX_PIECES_NUM + (0 if white else 5) \
                                        + piece_indices[char]] += 1

                black_offset = NUM_PIECES_PER_SIDE * NUM_SLOTS_PER_PIECE
                # Get the current gf index based on piece type and color (5 entries per piece)
                curr_index = S_IDX_PIECE_LIST \
                           + (0 if white else black_offset) \
                           + piece_desc_index[char] * NUM_SLOTS_PER_PIECE

                # print "piece: %s, white: %d, index: %d" % (char, white, curr_index)
                # Increment gf index if slot is already filled with an identical piece
                count = 1
                too_many_pieces = False
                while gf[curr_index] == 1:
                    count += 1
                    if char == 'k':
                        start_num_pieces = 1
                    elif char == 'q':
                        start_num_pieces = 1
                    elif char == 'p':
                        start_num_pieces = 8
                    else:
                        start_num_pieces = 2

                    if count > start_num_pieces:
                        too_many_pieces = True
                        break

                    curr_index += NUM_SLOTS_PER_PIECE

                if too_many_pieces:
                    continue
                # Mark piece as present
                gf[curr_index] = 1

                # Mark location
                gf[curr_index + 1] = c_rank
                gf[curr_index + 2] = c_file # TODO: Maybe normalize coordinates? They are normalized in Giraffe
                board_to_piece_index[(c_rank, c_file)] = curr_index
                board_to_piece_type[(c_rank, c_file)] = char
                # set occupied bitmap
                occupied_bitmap[c_rank][c_file] = 1 if white else -1                
            c_file += 1 # Increment file

    # Iterate through piece lists.
    for i in xrange(S_IDX_PIECE_LIST, S_IDX_SLIDE_LIST, NUM_SLOTS_PER_PIECE):
        # If piece is not present, skip
        if gf[i] == 0:
            continue

        # Fetch coordinate
        c_rank, c_file = gf[i + 1: i + 3]
        if 0 <= i - S_IDX_PIECE_LIST < \
            NUM_SLIDE_PIECES_PER_SIDE * NUM_SLOTS_PER_PIECE or \
            0 <= i - (S_IDX_PIECE_LIST + NUM_PIECES_PER_SIDE * NUM_SLOTS_PER_PIECE) \
            < NUM_SLIDE_PIECES_PER_SIDE * NUM_SLOTS_PER_PIECE:
            # if piece is queen, rook, or bishop then populate range of motion information and attack + defend map
            check_range_of_motion(c_rank, c_file, \
                board_to_piece_type[(c_rank, c_file)], occupied_bitmap, \
                piece_values[board_to_piece_type[(c_rank, c_file)]], \
                board_to_piece_index, piece_index_to_slide_index[i], \
                S_IDX_ATKDEF_MAP, gf)
        else:
            # if not then just populate attack and defend map
            set_att_def_map(c_rank, c_file, \
                board_to_piece_type[(c_rank, c_file)], occupied_bitmap, \
                piece_values[board_to_piece_type[(c_rank, c_file)]], \
                board_to_piece_index, S_IDX_ATKDEF_MAP, gf)

    return np.array(gf)

def get_diagonals(channels):
    """
        Retrieves and returns the diagonals from the board

        Ouput:
            Diagonals[ndarray]:
                12 Channels: 6 for each you and your opponents piece types
                Types in order are: Pawns, Rooks, Knights, Bishops, Queens, King
                First 6 channels are your pieces, last 6 are opponents.
                Each piece array has 10 diagonals with max size of 8 (shorter diagonals are 0 padded at the end)
                Diagonal ordering is a3 up, a6 down, a2 up, a7 down, a1 up, a8 down, b1 up, b8 down, c1 up, c8 down
    """
    diagonals = np.zeros((10, 8, hp['NUM_CHANNELS']))
    for i in xrange(hp['NUM_CHANNELS']):
        index = 0
        for o in xrange(-2,3):
            diag_up = np.diagonal(channels[:, :, i], offset=o)
            diag_down = np.diagonal(np.flipud(channels[:, :, i]), offset=o)

            diagonals[index, 0 : 8 - abs(o), i] = diag_up
            index += 1
            diagonals[index, 0 : 8 - abs(o), i] = diag_down
            index += 1

    return diagonals


def sigmoid_array(values):
    """ From: http://chesscomputer.tumblr.com/post/98632536555/using-the-stockfish-position-evaluation-score-to
        1000 cp lead almost guarantees a win (a sigmoid within that). From the looking at the graph to gather a few
        data points and using a sigmoid curve fitter an inaccurate function of 1/(1+e^(-0.00547x)) was decided on
        (by me, deal with it). Ideally this fitter function is learned, but this is just for testing so..."""
    return 1. / (1. + np.exp(-0.00547 * values))


def white_is_next(fen):
    """
    Returns true if fen is for white playing next.
    Inputs:
        fen [String]
            Chess board FEN.
    Output:
        [Boolean]
            True if fen is for white playing next.
    """
    if fen.split(' ')[1] == 'w':
        return True
    return False


def black_is_next(fen):
    """
    Returns true if fen is for black playing next.
    Inputs:
        fen [String]
            Chess board FEN.
    Output:
        [Boolean]
            True if fen is for black playing next.
    """
    return not white_is_next(fen)


def diff_dict_helper(old_dict, new_dict):
    """
    Compares two dictionaries of numpy.arrays. Useful for comparing weights and training variables.
    Input
        old_dict [Dictionary of numpy.arrays]
            One of the dictionary you'd like to compare.
        new_dict [Dictionary of numpy.arrays]
            The other dictionary you'd like to compare.
    Output:
        Result [None or String]
            Returns None if identical, otherwise returns error message.
    """

    for weight in old_dict.iterkeys():
        if isinstance(new_dict[weight], list) and isinstance(old_dict[weight], list):
            success = all([np.array_equal(old_dict[weight][i], new_dict[weight][i])
                           for i in range(len(old_dict[weight]))])
            success = success and (len(old_dict[weight]) == len(new_dict[weight]))
        elif type(new_dict[weight]) == type(old_dict[weight]):
            success = np.array_equal(np.array(old_dict[weight]), np.array(new_dict[weight]))
        else:
            success = False

        if not success:
            return "Mismatching entries for '%s': Expected:\n %s \n Received:\n %s\n" % (weight,
                                                                                           str(old_dict[weight]),
                                                                                str(new_dict[weight]))

    if len(old_dict) != len(new_dict):
        return "Different number of entries for '%s': Expected Length:\n %s \n Received Length:\n %s\n" % (weight,                                                                                                  len(old_dict),                                                                                                  len(new_dict))

    return None

def main():
    pass


if __name__ == '__main__':
    main()# White non-pawn piece position