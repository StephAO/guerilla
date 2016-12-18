import time
import chess
import numpy as np
from guerilla.hyper_parameters import *

flatten = lambda l: [item for sublist in l for item in sublist]

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

# TODO: deal with en passant and castling
def fen_to_bitmap(fen):
    """
        Converts a fen string to channels for neural net.
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

def check_range_of_motion(c_rank, c_file, piece, occupied_bitmap, piece_value, ps_index, ps):
    # forwards, backwards for each movement: vertical, horizontal, up diag, down diag
    still_sliding = [[True] * 2 for _ in xrange(4)]
    for i in xrange(1, 8):
        # left
        if (piece == 'q' or piece == 'r') and still_sliding[0][0] and c_file - i >= -1 and \
            (c_file - i == -1 or occupied_bitmap[c_rank][c_file - i] != 0):
            # Slide boundary
            still_sliding[0][0] = False
            ps[ps_index][0][0] = 1 - i
            # Defender/attacker map
            if c_file - i != -1:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank][c_file - i])
                ps[89][c_rank][c_file - i][0 if defender else 1] = min(piece_value, ps[89][c_rank][c_file - i][0 if defender else 1])
        # right
        if (piece == 'q' or piece == 'r') and still_sliding[0][1] and c_file + i <= 8 and \
            (c_file + i == 8 or occupied_bitmap[c_rank][c_file + i] != 0):
            # Slide boundary
            still_sliding[0][1] = False
            ps[ps_index][0][1] = i - 1
            # Defender/attacker map
            if c_file + i != 8:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank][c_file + i])
                ps[89][c_rank][c_file + i][0 if defender else 1] = min(piece_value, ps[89][c_rank][c_file + i][0 if defender else 1])
        # down
        if (piece == 'q' or piece == 'r') and still_sliding[1][0] and c_rank - i >= -1 and \
            (c_rank - i == -1 or occupied_bitmap[c_rank - i][c_file] != 0):
            # Slide boundary
            still_sliding[1][0] = False
            ps[ps_index][1][0] = 1 - i
            # Defender/attacker map
            if c_rank - i != -1:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank - i][c_file])
                ps[89][c_rank - i][c_file][0 if defender else 1] = min(piece_value, ps[89][c_rank - i][c_file][0 if defender else 1])
        # up
        if (piece == 'q' or piece == 'r') and still_sliding[1][1] and c_rank + i <= 8 and \
            (c_rank + i == 8 or occupied_bitmap[c_rank + i][c_file] != 0):
            # Slide boundary
            still_sliding[1][1] = False
            ps[ps_index][1][1] = i - 1
            # Defender/attacker map
            if c_rank + i != 8:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank + i][c_file])
                ps[89][c_rank + i][c_file][0 if defender else 1] = min(piece_value, ps[89][c_rank + i][c_file][0 if defender else 1])
        # left down
        if (piece == 'q' or piece == 'b') and still_sliding[2][0] and c_rank - i >= -1 and c_file - i >= -1 and \
            (c_rank - i == -1 or c_file - i == -1 or occupied_bitmap[c_rank - i][c_file - i] != 0):
            # Slide boundary
            still_sliding[2][0] = False
            ps[ps_index][0 if piece == 'b' else 2][0] = 1 - i
            # Defender/attacker map
            if c_rank - i != -1 and c_file - i != -1:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank - i][c_file - i])
                ps[89][c_rank - i][c_file - i][0 if defender else 1] = min(piece_value, ps[89][c_rank - i][c_file - i][0 if defender else 1])
        # right up
        if (piece == 'q' or piece == 'b') and still_sliding[2][1] and c_rank + i <= 8 and c_file + i <= 8 and \
            (c_rank + i == 8 or c_file + i == 8 or occupied_bitmap[c_rank + i][c_file + i] != 0):
            # Slide boundary
            still_sliding[2][1] = False
            ps[ps_index][0 if piece == 'b' else 2][1] = i - 1
            # Defender/attacker map
            if c_rank + i != 8 and c_file + i != 8:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank + i][c_file + i])
                ps[89][c_rank + i][c_file + i][0 if defender else 1] = min(piece_value, ps[89][c_rank + i][c_file + i][0 if defender else 1])
        # left up
        if (piece == 'q' or piece == 'b') and still_sliding[3][0] and c_rank + i <= 8 and c_file - i >= -1 and \
            (c_rank + i == 8 or c_file - i == -1 or occupied_bitmap[c_rank + i][c_file - i] != 0):
            # Slide boundary
            still_sliding[3][0] = False
            ps[ps_index][1 if piece == 'b' else 3][0] = 1 - i
            # Defender/attacker map
            if c_rank + i != 8 and c_file - i != -1:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank + i][c_file - i])
                ps[89][c_rank + i][c_file - i][0 if defender else 1] = min(piece_value, ps[89][c_rank + i][c_file - i][0 if defender else 1])
        # right down
        if (piece == 'q' or piece == 'b') and still_sliding[3][1] and c_rank - i >= -1 and c_file + i <= 8 and \
            (c_rank - i == -1 or c_file + i == 8 or occupied_bitmap[c_rank - i][c_file + i] != 0):
            # Slide boundary
            still_sliding[3][1] = False
            ps[ps_index][1 if piece == 'b' else 3][1] = i - 1
            # Defender/attacker map
            if c_rank - i != -1 and c_file + i != 8:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank - i][c_file + i])
                ps[89][c_rank - i][c_file + i][0 if defender else 1] = min(piece_value, ps[89][c_rank - i][c_file + i][0 if defender else 1])

def set_att_def_map(c_rank, c_file, piece, occupied_bitmap, piece_value, ps):

    if piece == 'n':
        knight_moves = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)]
        for i, j in knight_moves:
            if 0 <= c_rank + i < 8 and 0 <= c_file + j < 8 and occupied_bitmap[c_rank + i][c_file + j] != 0:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank + i][c_file + j])
                ps[89][c_rank + i][c_file + j][0 if defender else 1] = min(piece_value, ps[89][c_rank + i][c_file + j][0 if defender else 1])
        return

    white = (occupied_bitmap[c_rank][c_file] == 1)
    if piece == 'p':
        if white:
            # only attack moves
            pawn_moves = [(1, 1), (1, -1)]
        else:
            pawn_moves = [(-1, 1), (-1, -1)]
        for i, j in pawn_moves:
            if 0 <= c_rank + i < 8 and 0 <= c_file + j < 8 and occupied_bitmap[c_rank + i][c_file + j] != 0:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank + i][c_file + j])
                ps[89][c_rank + i][c_file + j][0 if defender else 1] = min(piece_value, ps[89][c_rank + i][c_file + j][0 if defender else 1])
        return

    if piece == 'k':
        king_moves = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
        for i, j in king_moves:
            if 0 <= c_rank + i < 8 and 0 <= c_file + j < 8 and occupied_bitmap[c_rank + i][c_file + j] != 0:
                defender = (occupied_bitmap[c_rank][c_file] == occupied_bitmap[c_rank + i][c_file + j])
                ps[89][c_rank + i][c_file + j][0 if defender else 1] = min(piece_value, ps[89][c_rank + i][c_file + j][0 if defender else 1])
        return

def fen_to_position_description(fen):

    piece_desc_index = {
        'q': 0,
        'r': 1,
        'b': 3,
        'n': 5,
        'p': 7,
        'k': 15
    }

    piece_values = {
        'q': 9,
        'r': 5,
        'b': 3,
        'n': 3,
        'p': 1,
        'k': 1000
    }

    piece_index_to_slide_index = {
        15 : 79,
        17 : 80,
        19 : 81,
        21 : 82,
        23 : 83,
        47 : 84,
        49 : 85,
        51 : 86,
        53 : 87,
        55 : 88,
    }

    ps = []
    ps += [0] * 15
    ps += [0, (0,0)] * 32
    ps += [[[0] * 2 for _1 in range(4)] for _2 in range(1)]
    ps += [[[0] * 2 for _1 in range(2)] for _2 in range(4)]
    ps += [[[0] * 2 for _1 in range(4)] for _2 in range(1)]
    ps += [[[0] * 2 for _1 in range(2)] for _2 in range(4)] # a = [[0]*2 for _ in xrange(2)]
    ps += [[[[999999] * 2 for _1 in range(8)] for _2 in range(8)]]

    fen = fen.split(' ')
    board_str = fen[0]
    turn = fen[1]
    castling = fen[2]
    # en_passant = fen[3]
    
    # Used for sliding and attack/defense maps
    occupied_bitmap = [[0 for _1 in range(8)] for _2 in range(8)]
    board_to_ps_index = {}
    board_to_piece_type = {}
    
    ps[0] = 1 if (turn == 'w') else 0
    ps[1] = 1 if ('Q' == castling) else 0
    ps[2] = 1 if ('K' == castling) else 0
    ps[3] = 1 if ('q' == castling) else 0
    ps[4] = 1 if ('k' == castling) else 0

    ranks = board_str.split('/')
    ranks.reverse()
    for c_rank, rank in enumerate(ranks):
        c_file = 0
        for char in rank:
            if char.isdigit():
                c_file += int(char) - 1
            else:    
                white = char.isupper()
                char = char.lower()
                if char != 'k':
                    ps[5 + (0 if white else 5) + piece_indices[char]] += 1
                curr_index = 15 + (0 if white else 32) + piece_desc_index[char] * 2
                # print "piece: %s, white: %d, index: %d" % (char, white, curr_index)
                while ps[curr_index] == 1:
                    curr_index += 2
                ps[curr_index] = 1
                ps[curr_index + 1] = (c_rank, c_file)
                board_to_ps_index[(c_rank, c_file)] = curr_index
                board_to_piece_type[(c_rank, c_file)] = char
                # set occupied bitmap
                occupied_bitmap[c_rank][c_file] = 1 if white else -1                
            c_file += 1


    for i in xrange(15, 78, 2):
        if ps[i] == 0:
            continue
        c_rank, c_file = ps[i + 1]
        # piece is queen, rook, or bishop
        if 15 <= i <= 23 or 47 <= i <= 55:
            check_range_of_motion(c_rank, c_file, board_to_piece_type[(c_rank, c_file)], occupied_bitmap, \
                piece_values[board_to_piece_type[(c_rank, c_file)]], piece_index_to_slide_index[i], ps)
        else:
            set_att_def_map(c_rank, c_file, board_to_piece_type[(c_rank, c_file)], occupied_bitmap, \
                piece_values[board_to_piece_type[(c_rank, c_file)]], ps)

    return ps

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
    # White = 1, Black = 0
    # True = 1, False = 0
    random_fen = "3q3r/2QR1n2/1PR2p1b/1k2p3/1P6/3pN3/1PB1pKp1/3N4 w - - 0 1"
    positions_description = fen_to_position_description(random_fen)
    if len(positions_description) != 90:
        print "Failure: Size of position description is incorrect"
        return False
    if positions_description[0] != 1: # White's turn
        print "Failure: Turn description is incorrect"
        return False
    if positions_description[1] != 0 or positions_description[2] != 0 \
        or positions_description[3] != 0 or positions_description[4] != 0: # Castling options
        print "Failure: Castling description is incorrect"
        return False
    # Order is alwasy queens, rooks, bishops, knigths, pawns
    if positions_description[5] != 1 \
        or positions_description[6] != 2 \
        or positions_description[7] != 1 \
        or positions_description[8] != 2 \
        or positions_description[9] != 3: # White pieces count
        print "Failure: White piece count is incorrect"
        return False
    if positions_description[10] != 1 \
        or positions_description[11] != 1 \
        or positions_description[12] != 1 \
        or positions_description[13] != 1 \
        or positions_description[14] != 5: # Black piece count
        print "Failure: Black piece count is incorrect"
        return False
    if positions_description[15] != 1 or positions_description[16] != (6, 2) \
        or positions_description[17] != 1 or positions_description[18] != (5, 2) \
        or positions_description[19] != 1 or positions_description[20] != (6, 3) \
        or positions_description[21] != 1 or positions_description[22] != (1, 2) \
        or positions_description[23] != 0 or positions_description[24] != (0, 0) \
        or positions_description[25] != 1 or positions_description[26] != (0, 3) \
        or positions_description[27] != 1 or positions_description[28] != (2, 4) \
        or positions_description[29] != 1 or positions_description[30] != (1, 1) \
        or positions_description[31] != 1 or positions_description[32] != (3, 1) \
        or positions_description[33] != 1 or positions_description[34] != (5, 1) \
        or positions_description[35] != 0 or positions_description[36] != (0, 0) \
        or positions_description[37] != 0 or positions_description[38] != (0, 0) \
        or positions_description[39] != 0 or positions_description[40] != (0, 0) \
        or positions_description[41] != 0 or positions_description[42] != (0, 0) \
        or positions_description[43] != 0 or positions_description[44] != (0, 0) \
        or positions_description[45] != 1 or positions_description[46] != (1, 5): # White piece position
        print "Failure: White piece position is incorrect"
        for idx, i in enumerate(positions_description):
            print idx, i
        return False
    if positions_description[47] != 1 or positions_description[48] != (7, 3) \
        or positions_description[49] != 1 or positions_description[50] != (7, 7) \
        or positions_description[51] != 0 or positions_description[52] != (0, 0) \
        or positions_description[53] != 1 or positions_description[54] != (5, 7) \
        or positions_description[55] != 0 or positions_description[56] != (0, 0) \
        or positions_description[57] != 1 or positions_description[58] != (6, 5) \
        or positions_description[59] != 0 or positions_description[60] != (0, 0) \
        or positions_description[61] != 1 or positions_description[62] != (1, 4) \
        or positions_description[63] != 1 or positions_description[64] != (1, 6) \
        or positions_description[65] != 1 or positions_description[66] != (2, 3) \
        or positions_description[67] != 1 or positions_description[68] != (4, 4) \
        or positions_description[67] != 1 or positions_description[70] != (5, 5) \
        or positions_description[71] != 0 or positions_description[72] != (0, 0) \
        or positions_description[73] != 0 or positions_description[74] != (0, 0) \
        or positions_description[75] != 0 or positions_description[76] != (0, 0) \
        or positions_description[77] != 1 or positions_description[78] != (4, 1): # Black piece position
        print "Failure: Black piece position is incorrect"
        return False
    # Sliding order:
    # Rank (left -ive, right +ive), File (down -ive, up +ive)
    # Up diag (left down -ive, right up +ive), down diag (left up -ive, right down +ive)
    # (For queens, rank/file before diagonals)
    if positions_description[79] != [[-2, 0], [0, 1], [0, 0], [-1, 1]] \
        or positions_description[80] != [[0, 2], [-3, 0]] \
        or positions_description[81] != [[0, 1], [-3, 0]] \
        or positions_description[82] != [[-1, 0], [-2, 0]] \
        or positions_description[83] != [[0, 0], [0, 0]]: # White piece sliding
        print "Failure: White piece sliding is incorrect"
        return False
    if positions_description[84] != [[-3,3], [0, 0], [0, 0], [0, 1]] \
        or positions_description[85] != [[-3, 0], [-1, 0]] \
        or positions_description[86] != [[0, 0], [0, 0]] \
        or positions_description[87] != [[-2, 0], [-2, 0]] \
        or positions_description[88] != [[0, 0], [0, 0]]: # Black piece sliding
        print "Failure: Black piece sliding is incorrect"
        for pd in positions_description[84:89]:
            print pd
        return False

    # Attacker/defender maps. 64 x 2d tuple (defender value, attacker value), where attacker
    # is the opposite color as piece on current square and defender is the same color
    # King value = 1000
    success = True
    for c_rank, ranks in enumerate(positions_description[89]): 
        for c_file, files in enumerate(ranks):
            if c_rank == 0 and c_file == 3:
                success &= (positions_description[89][c_rank][c_file] == [3,1])
            if c_rank == 1 and c_file == 1:
                success &= (positions_description[89][c_rank][c_file] == [3,999999])
            if c_rank == 1 and c_file == 2:
                success &= (positions_description[89][c_rank][c_file] == [3,1])
            if c_rank == 1 and c_file == 4:
                success &= (positions_description[89][c_rank][c_file] == [1,1000])
            if c_rank == 1 and c_file == 5:
                success &= (positions_description[89][c_rank][c_file] == [3,999999])
            if c_rank == 1 and c_file == 6:
                success &= (positions_description[89][c_rank][c_file] == [999999,3])
            if c_rank == 2 and c_file == 3:
                success &= (positions_description[89][c_rank][c_file] == [999999,3])
            if c_rank == 2 and c_file == 4:
                success &= (positions_description[89][c_rank][c_file] == [3,3])
            if c_rank == 3 and c_file == 1:
                success &= (positions_description[89][c_rank][c_file] == [999999,1000])
            if c_rank == 4 and c_file == 1:
                success &= (positions_description[89][c_rank][c_file] == [999999,999999])
            if c_rank == 4 and c_file == 4:
                success &= (positions_description[89][c_rank][c_file] == [1,9])
            if c_rank == 5 and c_file == 1:
                success &= (positions_description[89][c_rank][c_file] == [5,1000])
            if c_rank == 5 and c_file == 2:
                success &= (positions_description[89][c_rank][c_file] == [9,1000])
            if c_rank == 5 and c_file == 5:
                success &= (positions_description[89][c_rank][c_file] == [9,5])
            if c_rank == 5 and c_file == 7:
                success &= (positions_description[89][c_rank][c_file] == [3,999999])
            if c_rank == 6 and c_file == 2:
                success &= (positions_description[89][c_rank][c_file] == [1,9])
            if c_rank == 6 and c_file == 3:
                success &= (positions_description[89][c_rank][c_file] == [9,9])
            if c_rank == 6 and c_file == 5:
                success &= (positions_description[89][c_rank][c_file] == [999999,5])
            if c_rank == 7 and c_file == 3:
                success &= (positions_description[89][c_rank][c_file] == [3,5])
            if c_rank == 7 and c_file == 7:
                success &= (positions_description[89][c_rank][c_file] == [3,999999])
    if not success:
        print "Failure: Defender/Attacker map is incorrect"
        return False

    print "Test passed"

    return True


if __name__ == '__main__':
    main()# White non-pawn piece position