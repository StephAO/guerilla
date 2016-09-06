from abc import ABCMeta, abstractmethod, abstractproperty

class Player:
	__metaclass__ = ABCMeta

	@abstractproperty
	def name(self):
		return 'Error - you should never see this'
		
	@abstractmethod
	def get_move(self):
		return 'Error - you should never see this'


class Guerilla(Player):
	
	def __init__(self, _name):
		self._name = _name

	@property
	def name(self):
		return self._name


A.register(B)
b = B()
print b.name