# usage: python loss_2_excel.py loss_20161121-054613.py
from pkg_resources import resource_filename
import pickle
import sys
import os

def main():

	f_name = resource_filname('guerilla.train', 'pickles/' + sys.argv[1])
	loss = pickle.load(open(f_name, 'r'))
	
	for i in range(len(loss['loss'])):
		print '\t'.join(map(str, [i, loss['loss'][i], loss['train_loss'][i]]))
	
if __name__ == '__main__':
	main()