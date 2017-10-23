"""
Splits the KingBase chess dataset into individual games.
"""

import os

new_file_key = '[Event'
smallfile = None
i = 0
files = [f for f in os.listdir('.') if os.path.isfile(f) and os.path.splitext(f)[1] == '.pgn']
for f in files:
    with open(f, 'r') as bigfile:
        file_count = 0
        for line in bigfile:
            if line.split(' ')[0] == new_file_key:
                i += 1
                if smallfile:
                    smallfile.close()
                small_filename = 'single_game_pgns/pgn_{}.pgn'.format(i)
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()
