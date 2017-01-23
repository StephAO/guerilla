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
        actual = dh.get_diagonals(dh.fen_to_bitmap(fens[i], 12), 12)
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
        actual = dh.fen_to_bitmap(fens[i], 12)
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
    giraffe = dh.fen_to_giraffe(random_fen).tolist()
    if len(giraffe) != 351:
        print "Failure: Size of giraffe is incorrect"
        return False
    if giraffe[0] != 1: # White's turn
        print "Failure: Turn description is incorrect"
        return False
    if giraffe[1] != 0 or giraffe[2] != 0 \
        or giraffe[3] != 0 or giraffe[4] != 0: # Castling options
        print "Failure: Castling description is incorrect"
        return False
    # Order is always queens, rooks, bishops, knigths, pawns
    if giraffe[5] != 1 \
        or giraffe[6] != 2 \
        or giraffe[7] != 1 \
        or giraffe[8] != 2 \
        or giraffe[9] != 3: # White pieces count
        print "Failure: White piece count is incorrect"
        return False
    if giraffe[10] != 1 \
        or giraffe[11] != 1 \
        or giraffe[12] != 1 \
        or giraffe[13] != 1 \
        or giraffe[14] != 5: # Black piece count
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
    piece_type_index = {
        'wq' : 0,
        'wr' : 1,
        'wb' : 2,
        'wn' : 3,
        'wp' : 4,
        'wk' : 5,
        'bq' : 6,
        'br' : 7,
        'bb' : 8,
        'bn' : 9,
        'bp' : 10,
        'bk' : 11
    }

    piece_move_slice = {
        'wq' : slice(12, 14, None),
        'wr1' : slice(14, 16, None),
        'wr2' : slice(16, 18, None),
        'wb' : slice(18, 20, None),
        'wn1' : slice(20, 22, None),
        'wn2' : slice(22, 24, None),
        'wp1' : slice(24, 26, None),
        'wp2' : slice(26, 28, None),
        'wk' : slice(28, 30, None),
        'bq' : slice(30, 32, None),
        'br1' : slice(32, 34, None),
        'br2' : slice(34, 36, None),
        'bb' : slice(36, 38, None),
        'bn1' : slice(38, 40, None),
        'bn2' : slice(40, 42, None),
        'bp1' : slice(42, 44, None),
        'bp2' : slice(44, 46, None),
        'bk' :slice(46, 48, None)
    }

    random_fen = "3k1B2/8/4p3/2b3P1/bP6/2nN3K/3P2RR/1R3q2 w - - 0 1"

    board_state, move_map = dh.fen_to_movemap(random_fen)

    success = True

    if board_state.shape != (15,):
        print "Failure: shape of board info is incorrect"
        success = False 

    if not (board_state == np.array([1,0,0,0,0,0,3,1,1,3,1,0,2,1,1])).all():
        print "Failure: Info of board_state is incorrect"
        success = False 

    if move_map.shape != ((8, 8, 48)):
        print "Failure: shape of move_map is incorrect"
        success = False 

    move_map = move_map.tolist()

    if move_map[0][0][piece_move_slice['wr1']] != [0,1]:
        print move_map[0][0] 
        print "Failure: Info for tile 0,0 is incorrect"
        success = False

    if move_map[0][1][piece_type_index['wr']] != 1 and \
        move_map[0][1][piece_move_slice['bq']] != [0,5] and \
        move_map[0][1][piece_move_slice['bn1']] != [2,2]:
        print "Failure: Info for tile 0,1 is incorrect"
        success = False

    if move_map[0][2][piece_move_slice['wn1']] != [2,3] and \
        move_map[0][2][piece_move_slice['wr1']] != [0,1] and \
        move_map[0][2][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 0,2 is incorrect"
        success = False

    if move_map[0][3][piece_move_slice['wr1']] != [0,1] and \
        move_map[0][3][piece_move_slice['bn1']] != [2,2] and \
        move_map[0][3][piece_move_slice['bb']] != [3,0] and \
        move_map[0][3][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 1,5 is incorrect"
        success = False

    if move_map[0][4][piece_move_slice['wn1']] != [2,3] and \
        move_map[0][4][piece_move_slice['wr1']] != [0,1] and \
        move_map[0][4][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 0,4 is incorrect"
        print move_map[0][4]
        success = False

    if move_map[0][5][piece_type_index['bq']] != 1 and \
        move_map[0][5][piece_move_slice['wr1']] != [0,1]:
        print "Failure: Info for tile 0,5 is incorrect"
        success = False

    if move_map[0][6][piece_move_slice['wr2']] != [1,6] and \
        move_map[0][6][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 0,6 is incorrect"
        success = False

    if move_map[0][7][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 0,7 is incorrect"
        success = False

    if move_map[1][0][piece_move_slice['bn1']] != [2,2]:
        print "Failure: Info for tile 1,0 is incorrect"
        success = False

    if move_map[1][1][piece_move_slice['wr1']] != [0,1] and \
        move_map[1][1][piece_move_slice['wn1']] != [2,3]:
        print "Failure: Info for tile 1,1 is incorrect"
        success = False

    if move_map[1][2][piece_move_slice['bb']] != [3,0]:
        print "Failure: Info for tile 1,2 is incorrect"
        success = False

    if move_map[1][3][piece_type_index['wp']] != 1 and \
        move_map[1][3][piece_move_slice['wr2']] != [1,6]:
        print "Failure: Info for tile 1,3 is incorrect"
        success = False

    if move_map[1][4][piece_move_slice['wr2']] != [1,6] and \
        move_map[1][4][piece_move_slice['bn1']] != [2,2]:
        print "Failure: Info for tile 1,4 is incorrect"
        success = False

    if move_map[1][5][piece_move_slice['wr2']] != [1,6] and \
        move_map[1][5][piece_move_slice['wn1']] != [2,3]:
        print "Failure: Info for tile 1,5 is incorrect"
        success = False

    if move_map[1][6][piece_type_index['wr']] != 1 and \
        move_map[1][6][piece_move_slice['wk']] != [2,7] and \
        move_map[1][6][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 1,6 is incorrect"
        success = False

    if move_map[1][7][piece_move_slice['wr2']] != [1,6] and \
        move_map[1][7][piece_move_slice['wk']] != [2,7]:
        print "Failure: Info for tile 1,7 is incorrect"
        success = False

    if move_map[2][1][piece_move_slice['wr1']] != [0,1] and \
        move_map[2][1][piece_move_slice['bb']] != [3,0]:
        print "Failure: Info for tile 2,1 is incorrect"
        success = False

    if move_map[2][2][piece_type_index['bn']] != 1 and \
        move_map[2][2][piece_move_slice['wp1']] != [1,3]:
        print "Failure: Info for tile 2,2 is incorrect"
        success = False

    if move_map[2][3][piece_type_index['wn']] != 1 and \
        move_map[2][3][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 2,3 is incorrect"
        success = False

    if move_map[2][4][piece_move_slice['bb']] != [4,2] and \
        move_map[2][2][piece_move_slice['wp1']] != [1,3]:
        print "Failure: Info for tile 2,4 is incorrect"
        success = False

    if move_map[2][5][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 2,5 is incorrect"
        success = False

    if move_map[2][6][piece_move_slice['wr2']] != [1,6] and \
        move_map[2][6][piece_move_slice['wk']] != [2,7]:
        print "Failure: Info for tile 2,6 is incorrect"
        success = False

    if move_map[2][7][piece_type_index['wk']] != 1:
        print "Failure: Info for tile 2,7 is incorrect"
        success = False

    if move_map[3][0][piece_type_index['bb']] != 1 and \
        move_map[3][0][piece_move_slice['bn1']] != [2,2]:
        print "Failure: Info for tile 3,0 is incorrect"
        success = False

    if move_map[3][1][piece_type_index['wp']] != 1 and \
        move_map[3][1][piece_move_slice['wn1']] != [2,3] and \
        move_map[3][1][piece_move_slice['wr1']] != [0,1] and \
        move_map[3][1][piece_move_slice['bb']] != [4,2]:
        print "Failure: Info for tile 1,5 is incorrect"
        success = False

    if move_map[3][3][piece_move_slice['bb']] != [4,2]:
        print "Failure: Info for tile 3,3 is incorrect"
        success = False

    if move_map[3][4][piece_move_slice['bn1']] != [2,2]:
        print "Failure: Info for tile 3,4 is incorrect"
        success = False

    if move_map[3][5][piece_move_slice['wn1']] != [2,3] and \
        move_map[3][5][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 3,5 is incorrect"
        success = False

    if move_map[3][6][piece_move_slice['wr2']] != [1,6] and \
        move_map[3][6][piece_move_slice['wk']] != [2,7]:
        print "Failure: Info for tile 3,6 is incorrect"
        success = False

    if move_map[3][7][piece_move_slice['wk']] != [2,7]:
        print "Failure: Info for tile 3,7 is incorrect"
        success = False

    if move_map[4][0][piece_move_slice['wp2']] != [3,1]:
        print "Failure: Info for tile 4,0 is incorrect"
        success = False

    if move_map[4][1][piece_move_slice['bb']] != [3,0] and \
        move_map[4][1][piece_move_slice['bn1']] != [2,2]:
        print "Failure: Info for tile 4,1 is incorrect"
        success = False

    if move_map[4][2][piece_type_index['bb']] != 1 and \
        move_map[4][2][piece_move_slice['wp1']] != [3,1] and \
        move_map[4][2][piece_move_slice['wn1']] != [2,3] and \
        move_map[4][2][piece_move_slice['wb']] != [7,5]:
        print "Failure: Info for tile 1,5 is incorrect"
        success = False

    if move_map[4][3][piece_move_slice['bn1']] != [2,2] and \
        move_map[4][3][piece_move_slice['bp2']] != [5,4]:
        print "Failure: Info for tile 4,3 is incorrect"
        success = False

    if move_map[4][4][piece_move_slice['wn1']] != [2,3]:
        print "Failure: Info for tile 4,4 is incorrect"
        success = False

    if move_map[4][5][piece_move_slice['bq']] != [0,5] and \
        move_map[4][3][piece_move_slice['bp1']] != [5,4]:
        print "Failure: Info for tile 4,5 is incorrect"
        success = False

    if move_map[4][6][piece_type_index['wp']] != 1 and \
        move_map[4][6][piece_move_slice['wr2']] != [1,6]:
        print "Failure: Info for tile 4,6 is incorrect"
        success = False

    if move_map[5][1][piece_move_slice['bb']] != [4,2]:
        print "Failure: Info for tile 5,1 is incorrect"
        success = False

    if move_map[5][2][piece_move_slice['bb']] != [3,0]:
        print "Failure: Info for tile 5,2 is incorrect"
        success = False

    if move_map[5][3][piece_move_slice['wb']] != [7,5] and \
        move_map[5][3][piece_move_slice['bb']] != [4,2]:
        print "Failure: Info for tile 5,3 is incorrect"
        success = False

    if move_map[5][4][piece_type_index['bp']] != 1:
        print "Failure: Info for tile 5,4 is incorrect"
        success = False

    if move_map[5][5][piece_move_slice['bq']] != [0,5] and \
        move_map[5][5][piece_move_slice['wp1']] != [4,6]:
        print "Failure: Info for tile 5,5 is incorrect"
        success = False

    if move_map[5][7][piece_move_slice['wp1']] != [4,6]:
        print "Failure: Info for tile 5,7 is incorrect"
        success = False

    if move_map[6][0][piece_move_slice['bb']] != [4,2]:
        print "Failure: Info for tile 6,0 is incorrect"
        success = False

    if move_map[6][2][piece_move_slice['bk']] != [7,3]:
        print "Failure: Info for tile 6,2 is incorrect"
        success = False

    if move_map[6][3][piece_move_slice['bb']] != [3,0] and \
        move_map[6][3][piece_move_slice['bk']] != [7,3]:
        print "Failure: Info for tile 6,3 is incorrect"
        success = False
    

    if move_map[6][4][piece_move_slice['wb']] != [7,5] and \
        move_map[6][4][piece_move_slice['bb']] != [4,2] and \
        move_map[6][4][piece_move_slice['bk']] != [7,3]:
        print "Failure: Info for tile 6,4 is incorrect"
        success = False
    

    if move_map[6][5][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 6,5 is incorrect"
        success = False

    if move_map[6][6][piece_move_slice['wb']] != [7,5]:
        print "Failure: Info for tile 6,6 is incorrect"
        success = False

    if move_map[7][2][piece_move_slice['bk']] != [7,3]:
        print "Failure: Info for tile 7,2 is incorrect"
        success = False

    if move_map[7][3][piece_type_index['bk']] != 1:
        print "Failure: Info for tile 7,3 is incorrect"
        success = False

    if move_map[7][4][piece_move_slice['bb']] != [3,0] and \
        move_map[7][4][piece_move_slice['bk']] != [7,3]:
        print "Failure: Info for tile 7,4 is incorrect"
        success = False

    if move_map[7][5][piece_type_index['wb']] != 1 and \
        move_map[7][5][piece_move_slice['bb']] != [4,2] and \
        move_map[7][5][piece_move_slice['bq']] != [0,5]:
        print "Failure: Info for tile 7,5 is incorrect"
        success = False

    return success

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