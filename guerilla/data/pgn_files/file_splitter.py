"""
Splits the KingBase chess dataset into individual games.
"""

new_file_key = '[Event'
smallfile = None
i = 0
with open('KingBaseLite2016-03-pgn/KingBaseLite2016-03-A00-A39.pgn', 'r') as bigfile:
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
