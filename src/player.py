from abc import ABCMeta, abstractmethod, abstractproperty

class Player:
    __metaclass__ = ABCMeta
        
    def __init__(self, name, colour):
        self.name = name
        self.colour = colour

    @abstractmethod
    def get_move(self, board):
        raise NotImplementedError("You should never see this")
        return NotImplemented
