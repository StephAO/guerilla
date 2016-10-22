import time
import os

dir_path = dir_path = os.path.dirname(os.path.abspath(__file__))

start_num = 0
if os.path.isfile(dir_path + '/../pickles/start_num.txt'):
    with open(dir_path + '/../pickles/start_num.txt', 'r') as f:
        l = f.readline()
        start_num = int(l)

i = start_num
with open(dir_path + '/../pickles/numbers.csv', 'a') as f:
	stime = time.clock()
	while time.clock() - stime < 2:
		f.write(str(i) + ',')
		i += 1

with open(dir_path + '/../pickles/start_num.txt', 'w') as f:
	f.write(str(i))