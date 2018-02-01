import numpy as np

# WIN/LOSE/DRAW CONSTANTS
WIN_VALUE = 20000
LOSE_VALUE = -20000
DRAW_VALUE = 0

# Data handler constants + Variables
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

BOARD_LENGTH = 8
BOARD_SIZE = 64
NUM_PIECE_TYPE = len(piece_indices) * 2

STATE_DATA_SIZE = 14
BOARD_DATA_SIZE = 128
PIECE_DATA_SIZE = 208
GF_FULL_SIZE = 350

MOVEMAP_TILE_SIZE = 30
BITMAP_TILE_SIZE = NUM_PIECE_TYPE

# create list of all possible piece movements
crosswise_fn = [
    lambda x: np.array([0, -x - 1]),  # left
    lambda x: np.array([0, +x + 1]),  # right
    lambda x: np.array([-x - 1, 0]),  # down
    lambda x: np.array([+x + 1, 0])  # up
]

crosswise = []
for fn in crosswise_fn:
    crosswise.extend([fn(i) for i in xrange(0, BOARD_LENGTH - 1)])

diagonals_fn = [
    lambda x: np.array([-x - 1, -x - 1]),  # down left
    lambda x: np.array([+x + 1, +x + 1]),  # up right
    lambda x: np.array([+x + 1, -x - 1]),  # up left
    lambda x: np.array([-x - 1, +x + 1])  # down right
]

diagonals = []
for fn in diagonals_fn:
    diagonals.extend([fn(i) for i in xrange(0, BOARD_LENGTH - 1)])

knight_moves = [np.array(x) for x in [[1, 2], [2, 1], [2, -1], [1, -2],
                                      [-1, -2], [-2, -1], [-2, 1], [-1, 2]]]
pawn_moves = [np.array(x) for x in [[1, 1], [1, -1]]]
king_moves = [np.array(x) for x in [[1, 0], [1, 1], [0, 1], [-1, 1],
                                    [-1, 0], [-1, -1], [0, -1], [1, -1]]]

piece_moves = {
    'q': crosswise + diagonals,
    'r': crosswise,
    'b': diagonals,
    'n': knight_moves,
    'p': pawn_moves,
    'k': king_moves
}

# Movemap Global Variables for testing

piece_type_index = {
    'wq': 0,
    'wr': 1,
    'wb': 2,
    'wn': 3,
    'wp': 4,
    'wk': 5,
    'bq': 6,
    'br': 7,
    'bb': 8,
    'bn': 9,
    'bp': 10,
    'bk': 11
}

piece_move_idx = {
    'wq': 12,
    'wr1': 13,
    'wr2': 14,
    'wb': 15,
    'wn1': 16,
    'wn2': 17,
    'wp1': 18,
    'wp2': 19,
    'wk': 20,
    'bq': 21,
    'br1': 22,
    'br2': 23,
    'bb': 24,
    'bn1': 25,
    'bn2': 26,
    'bp1': 27,
    'bp2': 28,
    'bk': 29
}

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


# TODO: Documentation
def flip_move(move):
    """
    Flips the input UCI move.
    :return: 
    """

    def flip_num(n):
        return str(9 - int(n))

    return move[0] + flip_num(move[1]) + move[2] + flip_num(move[3])

def fen_to_nn_input(fen, nn_type):
    """
        Return neural net input types base on hyper parameters

        Inputs:
            fen[string]:
                fen of board to be converted
        Outputs:
            nn_input[varies]:
                correct neural net inputs type
    """

    key = 'fen_to_' + nn_type
    global_vars = globals()
    if key in global_vars:
        return global_vars['fen_to_' + nn_type](fen)

    raise NotImplementedError("Error: No fen_to_%s function exists." % (nn_type))

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
    channels = np.zeros((BOARD_LENGTH, BOARD_LENGTH, BITMAP_TILE_SIZE))

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
    return (channels,)


def in_bounds(rank_idx, file_idx):
    """ Returns True if within the chess board boundary, False otherwise """
    # NOTE: Assumes 0 indexed
    return (0 <= rank_idx < BOARD_LENGTH) and (0 <= file_idx < BOARD_LENGTH)


def check_range_of_motion(c_rank, c_file, piece, occupied_bitmap, piece_value, \
                          board_to_piece_index, slide_index, \
                          piece_data, board_data):
    """
        Finds the range of motion of a sliding piece (Queen, Rook, or Bishop).
        Set slide range for piece for giraffe input data structure.
        Updates attack defend maps.

        Inputs:
            c_rank[int]:
                rank of piece (0-7)
            c_file[int]:
                file of piece (0-7)
            piece[char]:
                piece type ('q', 'r', 'b')
            occupied bitmap[int[8][8]]:
                bitmap of occupied tiles. white = 1, black = -1, empty = 0
            piece_value[int]:
                piece value
            board_to_piece_index[dict]:
                maps rank, file to index of piece in piece list of giraffe input
            slide_index[int]:
                index of pieces slide list of giraffe input
            piece_data[list]:
                piece data input
            board_data[list]:
                board data input
    """
    # rank (left, right), file (down, up), '/' diag (down-left, up-right) , '\' diag (up-left, down-right)
    still_sliding = [True] * 8 if piece == 'q' else [True] * 4

    directions = []
    if (piece == 'q' or piece == 'r'):
        directions += crosswise_fn
    if (piece == 'q' or piece == 'b'):
        directions += diagonals_fn

    pos = np.array([c_rank, c_file])

    for offset in xrange(0, BOARD_LENGTH):
        # check horizontal and vertical sliding
        for i, direction in enumerate(directions):
            # tile to check
            r, f = pos + direction(offset)
            # If end of slide (piece in the way or out of bounds)
            if still_sliding[i] \
                    and (not in_bounds(r, f) or occupied_bitmap[r][f] != 0):
                still_sliding[i] = False
                piece_data[slide_index + i] = offset
                # If stopped by a piece, set defender/attacker map
                if in_bounds(r, f):
                    defender = (occupied_bitmap[c_rank][c_file] \
                                == occupied_bitmap[r][f])
                    map_index = (r * BOARD_LENGTH + f) * 2
                    map_index += 0 if defender else 1
                    board_data[map_index] = min(piece_value, board_data[map_index])
                    piece_data[board_to_piece_index[(r, f)] + (3 if defender else 4)] = \
                        board_data[map_index]

            if not any(still_sliding):
                break


def set_att_def_map(c_rank, c_file, piece, occupied_bitmap, piece_value, \
                    board_to_piece_index, piece_data, board_data):
    """
        Updates attack defend maps for non sliding pieces.

        Inputs:
            c_rank[int]:
                rank of piece (0-7)
            c_file[int]:
                file of piece (0-7)
            piece[char]:
                piece type ('n', 'p', 'k')
            occupied bitmap[int[8][8]]:
                bitmap of occupied tiles. white = 1, black = -1, empty = 0
            piece_value[int]:
                piece value
            board_to_piece_index[dict]:
                maps rank, file to index of piece in piece list of giraffe input
            piece_data[list]:
                piece data input
            board_data[list]:
                board data input
    """

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
            map_index = (r * BOARD_LENGTH + f) * 2
            map_index += 0 if defender else 1
            board_data[map_index] = min(piece_value, board_data[map_index])
            piece_data[board_to_piece_index[(r, f)] + (3 if defender else 4)] = \
                board_data[map_index]


def fen_to_giraffe(fen):
    """
        Converts a fen string to giraffe input list for neural net.
        Giraffe input list is based on Matthew Lai's giraffe model.
        He describes his input on page 17 of https://arxiv.org/pdf/1509.01549v1.pdf.
        Due to insufficient information, ours version of giraffe is not identical
        to his (his length is 363, while ours is 351).

        Inputs:
            fen[string]:
                fen string describing current state. 

        Outputs:
            State data [list(15)]:
                Turn, Castling, num of pieces
            Board data [list(128)]:
                Attack and Defend maps
            Piece data [list(208)]:
                Piece exists, placement, lowest and highest valued attacker. 
                Slide range for sliding pieces
    """
    piece_desc_index = {
        'q': 0,
        'r': 1,
        'b': 3,
        'n': 5,
        'p': 7,
        'k': 15
    }

    piece_index_to_slide_index = {
        0: 160,
        5: 168,
        10: 172,
        15: 176,
        20: 180,
        80: 184,
        85: 192,
        90: 196,
        95: 200,
        100: 204,
    }

    # Start index for the material configuration
    START_MAT_CONF = 4
    SLIDE_LIST_SIZE = 48
    NUM_SLOTS_PER_PIECE = 5  # For piece list
    NUM_SLIDE_PIECES_PER_SIDE = 5  # queen + 2 rooks + 2 bishops
    NUM_PIECES_PER_SIDE = 16  # PER SIDE
    BLACK_SLIDE_OFFSET = NUM_PIECES_PER_SIDE * NUM_SLOTS_PER_PIECE

    state_data = [0] * STATE_DATA_SIZE
    board_data = [999999] * (BOARD_SIZE * 2)
    piece_data = []
    for i in xrange(NUM_PIECES_PER_SIDE * 2):
        piece_data += [0, 0, 0, 999999, 999999]
    piece_data += [0] * (SLIDE_LIST_SIZE)
    

    fen = fen.split(' ')
    board_str = fen[0]
    turn = fen[1]
    castling = fen[2]
    # en_passant = fen[3]

    # Used for sliding and attack/defense maps
    # +1 if white piece, -1 if black piece, 0 o/w
    occupied_bitmap = [[0] * BOARD_LENGTH for _ in range(BOARD_LENGTH)]
    board_to_piece_index = {}  # Key: Coordinate, Value: Piece location in piece_list
    board_to_piece_type = {}  # Key: Coordinate, Value: Piece type

    # Castling rights
    state_data[0] = 1 if ('Q' in castling) else 0
    state_data[1] = 1 if ('K' in castling) else 0
    state_data[2] = 1 if ('q' in castling) else 0
    state_data[3] = 1 if ('k' in castling) else 0

    # Iterate through ranks starting from rank 1
    ranks = board_str.split('/')
    ranks.reverse()
    for c_rank, rank in enumerate(ranks):
        c_file = 0  # File count
        for char in rank:
            if char.isdigit():
                # Increment file count when empty squares are encountered
                c_file += int(char) - 1
            else:
                white = char.isupper()
                char = char.lower()

                # Update material configuration
                if char != 'k':
                    state_data[START_MAT_CONF + (0 if white else 5) \
                       + piece_indices[char]] += 1

                # Get the current piece data index based on piece type and color (5 entries per piece)
                curr_index = (0 if white else BLACK_SLIDE_OFFSET) \
                             + piece_desc_index[char] * NUM_SLOTS_PER_PIECE

                # print "piece: %s, white: %d, index: %d" % (char, white, curr_index)
                # Increment piece data index if slot is already filled with an identical piece
                count = 1
                too_many_pieces = False
                while piece_data[curr_index] == 1:
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
                piece_data[curr_index] = 1

                # Mark location
                piece_data[curr_index + 1] = c_rank
                piece_data[curr_index + 2] = c_file  # TODO: Maybe normalize coordinates? They are normalized in Giraffe
                board_to_piece_index[(c_rank, c_file)] = curr_index
                board_to_piece_type[(c_rank, c_file)] = char
                # set occupied bitmap
                occupied_bitmap[c_rank][c_file] = 1 if white else -1
            c_file += 1  # Increment file

    # Iterate through piece lists.
    for i in xrange(NUM_PIECES_PER_SIDE * 2):
        idx = i * NUM_SLOTS_PER_PIECE
        # If piece is not present, skip
        if piece_data[idx] == 0:
            continue

        # Fetch coordinate
        c_rank, c_file = piece_data[idx + 1: idx + 3]
        if idx < NUM_SLIDE_PIECES_PER_SIDE * NUM_SLOTS_PER_PIECE or \
           0 <= idx - (BLACK_SLIDE_OFFSET) < NUM_SLIDE_PIECES_PER_SIDE * NUM_SLOTS_PER_PIECE:
            # if piece is queen, rook, or bishop (sliding piece) then populate range of motion information and attack + defend map
            check_range_of_motion(c_rank, c_file, \
                                  board_to_piece_type[(c_rank, c_file)], occupied_bitmap, \
                                  piece_values[board_to_piece_type[(c_rank, c_file)]], \
                                  board_to_piece_index, piece_index_to_slide_index[idx], \
                                  piece_data, board_data)
        else:
            # if not then just populate attack and defend map
            set_att_def_map(c_rank, c_file, \
                            board_to_piece_type[(c_rank, c_file)], occupied_bitmap, \
                            piece_values[board_to_piece_type[(c_rank, c_file)]], \
                            board_to_piece_index, piece_data, board_data)

    return np.array(state_data), np.array(board_data), np.array(piece_data)


def set_move_map(c_rank, c_file, piece, occupied_bitmap, piece_move_idx, mm):
    """
        Finds the range of motion of a sliding piece (Queen, Rook, or Bishop).

        Inputs:
            c_rank[int]:
                rank of piece (0-7)
            c_file[int]:
                file of piece (0-7)
            piece[String]:
                piece type (e.g. 'wq', 'br1', 'bk', ... see piece_move_idx)
            occupied bitmap[String[8][8]]:
                bitmap of occupied tiles. each tile has a piece (see above)
            piece_move_idx[Dict]:
                slice of move_map tile list for a given piece type
            mm[list]:
                move map input
    """
    piece_colour = piece[0]
    piece_type = piece[1]
    pos = np.array([c_rank, c_file])
    still_sliding = True
    i = 0

    while i < (len(piece_moves[piece_type])):
        move = piece_moves[piece_type][i]

        if piece_type == 'p' and piece_colour == 'b':
            move = [move[0] * (-1), move[1]]

        r, f = pos + move
        # Out of bounds
        if not in_bounds(r, f):
            still_sliding = False
        else:

            if piece_type == 'p':
                full_piece = piece + ('1' if f > c_file else '2')
            else:
                full_piece = piece

            # Set map
            # TODO Remove assert
            # assert(mm[r][f][piece_move_idx[full_piece]] != 1)
            mm[r][f][piece_move_idx[full_piece]] = 1

            # End of slide (piece in the way)
            if occupied_bitmap[r][f] != 0:
                still_sliding = False

        if piece_type in ['q', 'r', 'b'] and not still_sliding:
            i += (BOARD_LENGTH - 1) - (i % (BOARD_LENGTH - 1))
            still_sliding = True
        else:
            i += 1


def fen_to_movemap(fen):
    """ 
        Move map is a 64 x (12 + 9 + 9) representation of the board. Each
        tile (64) on the board contains:
            1. One hot encoding of the piece type (12)
            2. White pieces that can move to that square (9)
            3. Black pieces that can move to that square (9)
        One hot encoding order is wq, wr, wb, wn, wp, wk, bq, br, bb, bn, bp, bk
        9 is chosen because a piece can only be attacked/defended by 9 pieces
        at a time without having had a pawn promotion. Each piece is defined by
        its presence as an attacking or defending piece (1 if present, 0 if not present)
    
        Order for pieces attacking is 
        wq, wr*2, wb, wn*2, wp*2, wk, bq, br*2, bb*2, bn, bp*2, bk

        Board state is based on Giraffe and is an array of size 14:
            [0 - 3] Castling Rights
            [4 - 13] Number of each type of pieces

        Inputs:
            fen[String]:
                fen

        Outputs:
            bs [Numpy.Array]
                Board state information.
            mm [Numpy.Array]
                Move map information.
    """
    bs = np.zeros((14))
    mm = np.zeros((BOARD_LENGTH, BOARD_LENGTH, MOVEMAP_TILE_SIZE))

    # Start index for the material configuration
    START_MAT_CONF = 4

    fen = fen.split(' ')
    board_str = fen[0]
    castling = fen[2]

    piece_count = {
        'wq': 1,
        'wr': 2,
        'wb': 2,
        'wn': 2,
        'wp': 8,
        'wk': 1,
        'bq': 1,
        'br': 2,
        'bb': 2,
        'bn': 2,
        'bp': 8,
        'bk': 1
    }

    # Castling rights
    bs[0] = 1 if ('Q' in castling) else 0
    bs[1] = 1 if ('K' in castling) else 0
    bs[2] = 1 if ('q' in castling) else 0
    bs[3] = 1 if ('k' in castling) else 0

    occupied_bitmap = [[0] * BOARD_LENGTH for _ in range(BOARD_LENGTH)]

    ranks = board_str.split('/')
    ranks.reverse()
    for c_rank, rank in enumerate(ranks):
        c_file = 0  # File count
        for char in rank:
            if char.isdigit():
                # Increment file count when empty squares are encountered
                c_file += int(char) - 1
            else:
                white = char.isupper()
                char = char.lower()
                piece = ('w' if white else 'b') + char

                # Increment piece count if not king
                if char != 'k':
                    bs[START_MAT_CONF + (0 if white else 5) \
                       + piece_indices[char]] += 1

                # One hot encoding of piece occupying tile
                mm[c_rank][c_file][piece_type_index[piece]] = 1

                piece_count[piece] -= 1
                if piece_count[piece] < 0:
                    # Handles promotions where you have more than the normal number of pieces.
                    #   Skipping the population of movemap for extra pieces is necessary due to a set number of slots.
                    #   This step marks it as 'x'-tra so that it can be skipped later
                    piece += 'x'
                elif char in ['r', 'n']:
                    # Label rook and knights based on rank + file (since a piece can be reached by multiple r/n)
                    #   Use for correct slot assingment in set_move_map
                    piece += str(2 - piece_count[piece])

                occupied_bitmap[c_rank][c_file] = piece

            c_file += 1
            if c_file > BOARD_LENGTH:
                raise ValueError("Fen has more than 8 pieces on a single rank")

    for c_rank in xrange(BOARD_LENGTH):
        for c_file in xrange(BOARD_LENGTH):
            p_info = occupied_bitmap[c_rank][c_file]
            if p_info != 0 and (len(p_info) < 3 or p_info[2] !='x'):
                # There exists a piece, and it isn't extra
                piece = occupied_bitmap[c_rank][c_file]
                set_move_map(c_rank, c_file, piece, occupied_bitmap,
                             piece_move_idx, mm)

    return bs, mm


def get_diagonals(channels, size_per_tile):
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
    diagonals = np.zeros((10, BOARD_LENGTH, size_per_tile))
    for i in xrange(size_per_tile):
        index = 0
        for o in xrange(-2, 3):
            diag_up = np.diagonal(channels[:, :, i], offset=o)
            diag_down = np.diagonal(np.flipud(channels[:, :, i]), offset=o)

            diagonals[index, 0: BOARD_LENGTH - abs(o), i] = diag_up
            index += 1
            diagonals[index, 0: BOARD_LENGTH - abs(o), i] = diag_down
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


def strip_fen(fen, keep_idxs=0):
    """
    Splits the input fen by space character, joins by a space the items noted by keep_idxs and returns as a string.
    Inputs:
        fen [String]
            Chess board FEN.
        keep_idxs [List of Ints] (Optional)
            Parts of the fen to keep once it has been split. By default keeps only the board state (index 0).
            Order of recombined list is order of keep_idxs.
    Output:
        [String]
            Stripped fen.
    """

    if not isinstance(keep_idxs, list):
        keep_idxs = [keep_idxs]

    if any([x > 5 for x in keep_idxs]):
        raise ValueError('All keep_idxs must be <= 5')

    fen_split = fen.split(' ')

    return (' ').join([fen_split[i] for i in keep_idxs])


def flip_to_white(fen):
    """
    Flips the fen to white playing next if necessary.
    Inputs:
        fen [String]
            Chess board FEN.
    Output:
        [String]
            Output FEN
    """
    if black_is_next(fen):
        return flip_board(fen)
    return fen

def diff_dict_helper(list_of_dicts):
    """
    Compares two dictionaries of numpy.arrays. Useful for comparing weights and training variables.
    Input
        old_dict [Dictionary of numpy.arrays]
            List of dictionaries you would like to compare.
    Output:
        Result [None or String]
            Returns None if identical, otherwise returns error message.
    """

    for i in range(len(list_of_dicts) - 1):
        old_dict = list_of_dicts[i]
        new_dict = list_of_dicts[i + 1]
        for weight in old_dict.iterkeys():
            if isinstance(new_dict[weight], list) and isinstance(old_dict[weight], list):
                success = all([
                    np.array_equal(old_dict[weight][j], new_dict[weight][j])
                    for j in range(len(old_dict[weight]))
                ])
                success = success and (len(old_dict[weight]) == len(new_dict[weight]))
            elif type(new_dict[weight]) == type(old_dict[weight]):
                success = np.array_equal(np.array(old_dict[weight]), np.array(new_dict[weight]))
            else:
                success = False

            if not success:
                return "Mismatching entries for dicts (%d, %d) key '%s': " \
                       "Expected:\n %s \n Received:\n %s\n" % (i, i + 1, weight,
                                                               str(old_dict[weight]), str(new_dict[weight]))

        if len(old_dict) != len(new_dict):
            return "Different number of entries for dicts (%d, %d) : First Length:\n %s \n Second Length:\n %s\n" % (
                i, i + 1, len(old_dict), len(new_dict))

    return None


def material_score(fen):
    """
    Returns the material score for the input FEN.
    Input:
        fen [String]
            FEN from which the material score is calculated.
    Output:
        scores [Dictionary]
            Output is of the form {'w': white material score, 'b': black material score}
    """

    fen = strip_fen(fen)
    scores = {'w': 0, 'b': 0}

    for c in fen:
        if c.isalpha():
            player = 'w' if c.isupper() else 'b'
            piece = c.lower()
            # Skip if king
            if piece != 'k':
                scores[player] += piece_values[piece]

    return scores

def main():
    a = fen_to_movemap("8/8/8/8/8/3k4/8/r2K4 w - - 0 1");
    print a[1][0][3]
    print a[1][0][0]


if __name__ == '__main__':
    main()
