import player
import neural_net
import search

class Guerilla(Player):
    
    def __init__(self, _name):
        self._name = _name
        # self.nn = Neu

    @property
    def name(self):
        return self._name

    def get_move(self, board):
        return 1

player.Player.register(Guerilla)
