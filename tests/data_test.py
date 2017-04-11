import chess
import numpy as np
from pkg_resources import resource_filename

import guerilla.data_handler as dh

###############################################################################
# Input Tests
###############################################################################

def diag_input_test():
    """
    Tests that boards are properly converted to diagonals.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    # Notes on diagonal structure:
    #   Diagonal order is upwards (/) then downwards (\) with respect to chess board (origin is bottom left)
    #   There are 10 diagonals in total. Shorter diagonals are 0 padded at the end.
    #   Diagonals vary in size from 6 to 8
    #   Diagonal ordering is a3 up, a6 down, a2 up, a7 down, a1 up, a8 down, b1 up, b8 down, c1 up, c8 down

    # Fens and Expected Results
    fens = []
    corr = []

    ### Randomly selected fen ###
    rand_fen = 'r2q1rk1/1pp2ppp/p1np1n2/1B2p3/Pb2P1b1/3P1N2/1PP1NPPP/R1BQ1RK1 w - - 0 9'

    # Expected result
    rand_channels = np.zeros([8, 8, 12])

    # Positions
    pos = {}

    # White Queen, Rook, Bishops, Knights, Pawns then King
    pos[0] = [(0, 3)]
    pos[1] = [(0, 0), (0, 5)]
    pos[2] = [(4, 1), (0, 2)]
    pos[3] = [(2, 5), (1, 4)]
    pos[4] = [(3, 0), (3, 4), (2, 3), (1, 1), (1, 2), (1, 5), (1, 6), (1, 7)]
    pos[5] = [(0, 6)]

    # Black Pawn, Rook, Knights, Bishops, Queens then King
    pos[6] = [(7, 3)]
    pos[7] = [(7, 0), (7, 5)]
    pos[8] = [(3, 1), (3, 6)]
    pos[9] = [(5, 2), (5, 5)]
    pos[10] = [(6, 1), (6, 2), (6, 5), (6, 6), (6, 7), (5, 0), (5, 3), (4, 4)]
    pos[11] = [(7, 6)]

    for channel, loc in pos.iteritems():
        for rank, file in loc:
            rand_channels[rank][file][channel] = 1

    # Convert to diagonals
    rand_corr = np.zeros([10, 8, 12])
    for channel in range(rand_channels.shape[2]):
        idx = 0
        for i in range(-2, 3):
            unpad_up = np.diagonal(rand_channels[:, :, channel], offset=i)
            unpad_down = np.diagonal(np.flipud(rand_channels[:, :, channel]), offset=i)
            rand_corr[2 * idx, :, channel] = np.lib.pad(unpad_up, (0, 8 - len(unpad_down)), 'constant',
                                                        constant_values=0)
            rand_corr[2 * idx + 1, :, channel] = np.lib.pad(unpad_down, (0, 8 - len(unpad_down)),
                                                            'constant', constant_values=0)
            idx += 1

    fens.append(rand_fen)
    corr.append(rand_corr)

    ### Completely filled board with each piece ###
    diag_vecs = [[1] * 6 + [0] * 2, [1] * 7 + [0], [1] * 8, [1] * 7 + [0], [1] * 6 + [0] * 2]
    filled_diag = np.repeat(np.stack(diag_vecs, axis=0), 2, axis=0)
    for i, piece in enumerate('QRBNPKqrbnpk'):
        fill_fen = '/'.join([piece * 8 for _ in range(8)]) + ' w KQkq - 0 1'
        fill_corr = np.dstack([np.zeros([10, 8, i]), filled_diag, np.zeros([10, 8, 11 - i])])
        fens.append(fill_fen)
        corr.append(fill_corr)

    ### Empty board ###
    empty_fen = '/'.join(['8'] * 8) + ' w KQkq - 0 1'
    empty_corr = np.zeros([10, 8, 12])
    fens.append(empty_fen)
    corr.append(empty_corr)

    success = True
    for i in range(len(fens)):
        actual = dh.get_diagonals(dh.fen_to_bitmap(fens[i])[0], 12)
        if not np.array_equal(corr[i], actual):
            print "Failed converting fen to diagonals:"
            print fens[i]
            print "Expected:"
            for j in range(12):
                print corr[i][:, :, j]
            print "Received:"
            for j in range(12):
                print actual[:, :, j]

                success = False

    return success


def bitmap_input_test():
    """
    Tests that boards are properly converted to bitmap channels.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    # Notes:
    #   Channel order is [q,r,b,n,p,k] white and then black
    #   Within a board the order is Rank 8 -> Rank 1 and File a -> File h

    # Fens and Expected Results
    fens = []
    corr = []

    ### Randomly selected fen ###
    rand_fen = 'r2q1rk1/1pp2ppp/p1np1n2/1B2p3/Pb2P1b1/3P1N2/1PP1NPPP/R1BQ1RK1 w - - 0 9'

    # Expected result
    rand_corr = np.zeros([8, 8, 12])

    # Positions
    pos = {}

    # White Queen, Rook, Bishops, Knights, Pawns then King
    pos[0] = [(0, 3)]
    pos[1] = [(0, 0), (0, 5)]
    pos[2] = [(4, 1), (0, 2)]
    pos[3] = [(2, 5), (1, 4)]
    pos[4] = [(3, 0), (3, 4), (2, 3), (1, 1), (1, 2), (1, 5), (1, 6), (1, 7)]
    pos[5] = [(0, 6)]

    # Black Queen, Rook, Bishops, Knights, Pawns then King
    pos[6] = [(7, 3)]
    pos[7] = [(7, 0), (7, 5)]
    pos[8] = [(3, 1), (3, 6)]
    pos[9] = [(5, 2), (5, 5)]
    pos[10] = [(6, 1), (6, 2), (6, 5), (6, 6), (6, 7), (5, 0), (5, 3), (4, 4)]
    pos[11] = [(7, 6)]

    for channel, loc in pos.iteritems():
        for rank, file in loc:
            rand_corr[rank][file][channel] = 1

    fens.append(rand_fen)
    corr.append(rand_corr)

    ### Completely filled board with each piece ###
    for i, piece in enumerate('QRBNPKqrbnpk'):
        fill_fen = '/'.join([piece * 8 for _ in range(8)]) + ' w KQkq - 0 1'
        fill_corr = np.dstack([np.zeros([8, 8, i]), np.ones([8, 8, 1]), np.zeros([8, 8, 11 - i])])
        fens.append(fill_fen)
        corr.append(fill_corr)

    ### Empty board ###
    empty_fen = '/'.join(['8'] * 8) + ' w KQkq - 0 1'
    empty_corr = np.zeros([8, 8, 12])
    fens.append(empty_fen)
    corr.append(empty_corr)

    success = True
    for i in range(len(fens)):
        actual = dh.fen_to_bitmap(fens[i])[0]
        if not np.array_equal(corr[i], actual):
            print "Failed converting fen to channels:"
            print fens[i]
            print "Expected:"
            for j in range(12):
                print corr[i][:, :, j]
            print "Received:"
            for j in range(12):
                print actual[:, :, j]

                success = False

    return success

def giraffe_input_test():
    """
    Tests that boards are properly converted to giraffe list.
    Output:
        Result [Boolean]
            True if test passed, False if test failed.
    """
    # White = 1, Black = 0
    # True = 1, False = 0
    random_fen = "3q3r/2QR1n2/1PR2p1b/1k2p3/1P6/3pN3/1PB1pKp1/3N4 w - - 0 1"
    state, board, piece = dh.fen_to_giraffe(random_fen)
    giraffe = []
    giraffe.extend(state.tolist())
    giraffe.append(None) # hacky way to fix removing turn to move
    giraffe.extend(piece.tolist())
    giraffe.extend(board.tolist())
    if len(giraffe) != 351:
        print "Failure: Size of giraffe is incorrect"
        return False
    if giraffe[0] != 0 or giraffe[1] != 0 \
        or giraffe[2] != 0 or giraffe[3] != 0: # Castling options
        print "Failure: Castling description is incorrect"
        return False
    # Order is always queens, rooks, bishops, knigths, pawns
    if giraffe[4] != 1 \
        or giraffe[5] != 2 \
        or giraffe[6] != 1 \
        or giraffe[7] != 2 \
        or giraffe[8] != 3: # White pieces count
        print "Failure: White piece count is incorrect"
        return False
    if giraffe[9] != 1 \
        or giraffe[10] != 1 \
        or giraffe[11] != 1 \
        or giraffe[12] != 1 \
        or giraffe[13] != 5: # Black piece count
        print "Failure: Black piece count is incorrect"
        return False
    # exists, rank, file, lowest valued defender, lowest valued attacker
    if giraffe[15:20] != [1, 6, 2, 1, 9] \
        or giraffe[20:25] != [1, 5, 2, 9, 1000] \
        or giraffe[25:30] != [1, 6, 3, 9, 9] \
        or giraffe[30:35] != [1, 1, 2, 3, 1] \
        or giraffe[35:40] != [0, 0, 0, 999999, 999999] \
        or giraffe[40:45] != [1, 0, 3, 3, 1] \
        or giraffe[45:50] != [1, 2, 4, 3, 3] \
        or giraffe[50:55] != [1, 1, 1, 3, 999999] \
        or giraffe[55:60] != [1, 3, 1, 999999, 1000] \
        or giraffe[60:65] != [1, 5, 1, 5, 1000] \
        or giraffe[65:70] != [0, 0, 0, 999999, 999999] \
        or giraffe[70:75] != [0, 0, 0, 999999, 999999] \
        or giraffe[75:80] != [0, 0, 0, 999999, 999999] \
        or giraffe[80:85] != [0, 0, 0, 999999, 999999] \
        or giraffe[85:90] != [0, 0, 0, 999999, 999999] \
        or giraffe[90:95] != [1, 1, 5, 3, 999999]: # White piece position
        print "Failure: White piece position is incorrect"
        return False
    if giraffe[95:100] != [1, 7, 3, 3, 5] \
        or giraffe[100:105] != [1, 7, 7, 3, 999999] \
        or giraffe[105:110] != [0, 0, 0, 999999, 999999] \
        or giraffe[110:115] != [1, 5, 7, 3, 999999] \
        or giraffe[115:120] != [0, 0, 0, 999999, 999999] \
        or giraffe[120:125] != [1, 6, 5, 999999, 5] \
        or giraffe[125:130] != [0, 0, 0, 999999, 999999] \
        or giraffe[130:135] != [1, 1, 4, 1, 1000] \
        or giraffe[135:140] != [1, 1, 6, 999999, 3] \
        or giraffe[140:145] != [1, 2, 3, 999999, 3] \
        or giraffe[145:150] != [1, 4, 4, 1, 9] \
        or giraffe[150:155] != [1, 5, 5, 9, 5] \
        or giraffe[155:160] != [0, 0, 0, 999999, 999999] \
        or giraffe[160:165] != [0, 0, 0, 999999, 999999] \
        or giraffe[165:170] != [0, 0, 0, 999999, 999999] \
        or giraffe[170:175] != [1, 4, 1, 999999, 999999]: # Black piece position
        print "Failure: Black piece position is incorrect"
        return False

    # Sliding order: rank (left, right), file (down, up), '/' diag (down-left, up-right) , '\' diag (up-left, down-right)
    # (For queens, rank/file before diagonals)
    if giraffe[175:183] != [2, 0, 0, 1, 0, 0, 1, 1] \
        or giraffe[183:187] != [0, 2, 3, 0] \
        or giraffe[187:191] != [0, 1, 3, 0] \
        or giraffe[191:195] != [1, 0, 2, 0] \
        or giraffe[195:199] != [0, 0, 0, 0]: # White piece sliding
        print "Failure: White piece sliding is incorrect"
        return False
    if giraffe[199:207] != [3, 3, 0, 0, 0, 0, 0, 1] \
        or giraffe[207:211] != [3, 0, 1, 0] \
        or giraffe[211:215] != [0, 0, 0, 0] \
        or giraffe[215:219] != [2, 0, 2, 0] \
        or giraffe[219:223] != [0, 0, 0, 0]: # Black piece sliding
        print "Failure: Black piece sliding is incorrect"
        return False

    # Defender/Attacker maps. 64 x 2 tuple (defender value, attacker value), where attacker
    # is the opposite color as piece on current square and defender is the same color
    # King value = 1000

    success = True
    for c_rank in xrange(8):
        for c_file in xrange(8):
            atk, dfn = giraffe[
                       223 + (c_rank * 8 + c_file) * 2 : \
                       223 + (c_rank * 8 + c_file) * 2 + 2]
            if c_rank == 0 and c_file == 3:
                success &= ([atk, dfn] == [3,1])
            elif c_rank == 1 and c_file == 1:
                success &= ([atk, dfn] == [3,999999])
            elif c_rank == 1 and c_file == 2:
                success &= ([atk, dfn] == [3,1])
            elif c_rank == 1 and c_file == 4:
                success &= ([atk, dfn] == [1,1000])
            elif c_rank == 1 and c_file == 5:
                success &= ([atk, dfn] == [3,999999])
            elif c_rank == 1 and c_file == 6:
                success &= ([atk, dfn] == [999999,3])
            elif c_rank == 2 and c_file == 3:
                success &= ([atk, dfn] == [999999,3])
            elif c_rank == 2 and c_file == 4:
                success &= ([atk, dfn] == [3,3])
            elif c_rank == 3 and c_file == 1:
                success &= ([atk, dfn] == [999999,1000])
            elif c_rank == 4 and c_file == 1:
                success &= ([atk, dfn] == [999999,999999])
            elif c_rank == 4 and c_file == 4:
                success &= ([atk, dfn] == [1,9])
            elif c_rank == 5 and c_file == 1:
                success &= ([atk, dfn] == [5,1000])
            elif c_rank == 5 and c_file == 2:
                success &= ([atk, dfn] == [9,1000])
            elif c_rank == 5 and c_file == 5:
                success &= ([atk, dfn] == [9,5])
            elif c_rank == 5 and c_file == 7:
                success &= ([atk, dfn] == [3,999999])
            elif c_rank == 6 and c_file == 2:
                success &= ([atk, dfn] == [1,9])
            elif c_rank == 6 and c_file == 3:
                success &= ([atk, dfn] == [9,9])
            elif c_rank == 6 and c_file == 5:
                success &= ([atk, dfn] == [999999,5])
            elif c_rank == 7 and c_file == 3:
                success &= ([atk, dfn] == [3,5])
            elif c_rank == 7 and c_file == 7:
                success &= ([atk, dfn] == [3,999999])
            else:
                success &= ([atk, dfn] == [999999, 999999])

    if not success:
        print "Failure: Defender/Attacker map is incorrect"
        return False
        
    return True

def move_map_input_test():

    random_fen = "3k1B2/8/4p3/2b3P1/bP6/2nN3K/3P2RR/1R3q2 w - - 0 1"

    board_state, move_map = dh.fen_to_movemap(random_fen)

    success = True

    if board_state.shape != (14,):
        print "Failure: shape of board info is incorrect"
        success = False 

    if not (board_state == np.array([0,0,0,0,0,3,1,1,3,1,0,2,1,1])).all():
        print "Failure: Info of board_state is incorrect"
        success = False 

    if move_map.shape != ((8, 8, 48)):
        print "Failure: shape of move_map is incorrect"
        success = False 

    move_map = move_map.tolist()

    correct = [(None, ('wr1', [1, 2])),  # Row 0
               ('wr', [('bq', [1, 6]), ('bn1', [3, 3])]),
               (None, [('wn1', [3, 4]), ('wr1', [1, 2]), ('bq', [1, 6])]),
               (None, [('wr1', [1, 2]), ('bn1', [3, 3]), ('bb', [4, 1]), ('bq', [1, 6])]),
               (None, [('wn1', [3, 4]), ('wr1', [1, 2]), ('bq', [1, 6])]),
               ('bq', [('wr1', [1, 2])]),
               (None, [('wr2', [2, 7]), ('bq', [1, 6]), ('bb', [5, 3])]),
               (None, [('bq', [1, 6])]),
               (None, [('bn1', [3, 3])]),  # Row 1
               (None, [('wr1', [1, 2]), ('wn1', [3, 4])]),
               (None, [('bb', [4, 1])]),
               ('wp', [('wr2', [2, 7])]),
               (None, [('wr2', [2, 7]), ('bn1', [3, 3]), ('bq', [1, 6])]),
               (None, [('wr2', [2, 7]), ('wn1', [3, 4]), ('bq', [1, 6]), ('bb', [5, 3])]),
               ('wr', [('wk', [3, 8]), ('bq', [1, 6])]),
               ('wr', [('wr2', [2, 7]), ('wk', [3, 8])]),
               (None, None),  # Row 2
               (None, [('wr1', [1, 2]), ('bb', [4, 1])]),
               ('bn', ('wp2', [2, 4])),
               ('wn', ('bq', [1, 6])),
               (None, [('bb', [5, 3]), ('wp1', [2, 4])]),
               (None, ('bq', [1, 6])),
               (None, [('wr2', [2, 7]), ('wk', [3, 8])]),
               ('wk', None),
               ('bb', ('bn1', [3, 3])),  # Row 3
               ('wp', [('wr1', [1, 2]), ('wn1', [3, 4]), ('bb', [5, 3])]),
               (None, None),
               (None, ('bb', [5, 3])),
               (None, ('bn1', [3, 3])),
               (None, [('wn1', [3, 4]), ('bq', [1, 6])]),
               (None, [('wr2', [2, 7]), ('wk', [3, 8])]),
               (None, ('wk', [3, 8])),
               (None, ('wp2', [4, 2])),  # Row 4
               (None, [('bb', [4, 1]), ('bn1', [3, 3])]),
               ('bb', [('wp1', [4, 2]), ('wn1', [3, 4]), ('wb', [8, 6])]),
               (None, [('bn1', [3, 3]), ('bp2', [6, 5])]),
               (None, ('wn1', [3, 4])),
               (None, [('bq', [1, 6]), ('bp1', [6, 5])]),
               ('wp', ('wr2', [2, 7])),
               (None, None),
               (None, None),
               (None, ('bb', [5, 3])),
               (None, ('bb', [4, 1])),
               (None, [('wb', [8, 6]), ('bb', [5, 3])]),
               ('bp', None),
               (None, [('bq', [1, 6]), ('wp2', [5, 7])]),
               (None, None),
               (None, [('wp1', [5, 7]), ('wb', [8, 6])]),
               (None, ('bb', [5, 3])),  # Row 6
               (None, None),
               (None, ('bk', [8, 4])),
               (None, [('bb', [4, 1]), ('bk', [8, 4])]),
               (None, [('wb', [8, 6]), ('bb', [5, 3]), ('bk', [8, 4])]),
               (None, ('bq', [1, 6])),
               (None, ('wb', [8, 6])),
               (None, None),
               (None, None),
               (None, None),
               (None, ('bk', [8, 4])),
               ('bk', None),
               (None, [('bb', [4, 1]), ('bk', [8, 4])]),
               ('wb', [('bb', [5, 3]), ('bq', [1, 6])]),
               (None, None),
               (None, None)
               ]

    for i in range(len(correct)):
        piece, move_pairs = correct[i]
        if not tile_check(move_map[i / 8][i % 8], piece, move_pairs):
            print "Failure: info for tile %d, %d is incorrect." % (i / 8, i % 8)
            success = False

    return success


def tile_check(tile, piece=None, move_pairs=None):
    """
    Helper for the movemap test. Checks the input tile movemap with the expected movemap.
    Input:
        tile [Array of Ints]
            Tile movemap.
        piece [String]
            The piece which occupies the square. If no piece then 'None'.
        move_pairs [List ofTuples]
            (Piece type, coordinate)
            All non-specified entries are set to 0.
    Output:
        [Boolean]
            True if tile movemap matches expected movemap.
    """

    if not move_pairs:
        move_pairs = []
    elif type(move_pairs) != list:
        move_pairs = [move_pairs]

    # Create 0 array + piece
    truth = [1 if (piece and i == dh.piece_type_index[piece]) else 0 for i in range(48)]

    # Attacking/defending pieces
    for piece, coordinates in move_pairs:
        truth[dh.piece_move_slice[piece]] = coordinates

    return (tile == truth)


def run_data_tests():
    all_tests = {}
    all_tests["Input Tests"] = {
        'Board to Bitmap': bitmap_input_test,
        'Bitmap to Diagonals': diag_input_test,
        'Board to Giraffe': giraffe_input_test,
        'Board to Move Map': move_map_input_test
    }

    success = True
    print "\nRunning Data Tests...\n"
    for group_name, group_dict in all_tests.iteritems():
        print "--- " + group_name + " ---"
        for name, test in group_dict.iteritems():
            print "Testing " + name + "..."
            if not test():
                print "%s test failed" % name.capitalize()
                success = False

    return success

def main():
    if run_data_tests():
        print "All tests passed"
    else:
        print "You broke something - go fix it"

if __name__ == '__main__':
    main()