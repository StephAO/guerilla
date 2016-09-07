from abc import ABCMeta, abstractmethod, abstractproperty

class Player:
    __metaclass__ = ABCMeta
        
    def __init__(self, name, colour = None):
        self.name = name
        self._colour = colour

    @property
    def colour(self):
        return self._colour

    @colour.setter
    def colour(self, colour):
        if colour.lower()  in ['white','w']:
            self._colour = 'white'
        elif colour.lower()  in ['black','b']:
            self._colour = 'black'
        else:
            raise ValueError("Error: Invalid colour! Must be 'white','w','black' or 'b'.")

    @abstractmethod
    def get_move(self, board):
        raise NotImplementedError("You should never see this")
        return NotImplemented
