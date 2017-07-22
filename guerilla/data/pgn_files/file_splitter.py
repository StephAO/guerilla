"""
Splits the KingBase chess dataset into individual games.
"""

import os

new_file_key = '[Event'
smallfile = None
i = 0
kb_path = 'D:/Datasets/Chess/KingBase2016-03-pgn'
for file in os.listdir(kb_path):
    fp = os.path.join(kb_path, file)
    if not os.path.isfile(fp):
        continue
    print file

    with open(fp, 'r') as bigfile:
        file_count = 0
        for line in bigfile:
            if line.split(' ')[0] == new_file_key:
                i += 1
                if smallfile:
                    smallfile.close()
                small_filename = os.path.join(kb_path,'../single_game_pgns/pgn_{}.pgn'.format(i))
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()
