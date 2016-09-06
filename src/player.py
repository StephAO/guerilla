from abc import ABCMeta, abstractmethod, abstractproperty

class Player:
	__metaclass__ = ABCMeta

	@abstractproperty
	def name(self):
		return 'Error - you should never see this'
		
	@abstractmethod
	def get_move(self, board):
		return 'Error - you should never see this'
