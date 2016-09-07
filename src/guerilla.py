import player
import neural_net
import search

class Guerilla(player.Player):
    
    def __init__(self, name, colour):
        super(Guerilla, self).__init__(name, colour)
        self.nn = neural_net.NeuralNet()
        self.search = search.Search(self.nn.evaluate)

    def get_move(self, board):
        return 1

player.Player.register(Guerilla)
