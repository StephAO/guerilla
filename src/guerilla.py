import player
import neural_net
import search

class Guerilla(player.Player):
    def __init__(self, name, colour=None, _load_file=None):
        super(Guerilla, self).__init__(name, colour)
        self.nn = neural_net.NeuralNet(load_weights=(_load_file is not None), load_file=_load_file)
        self.search = search.Search(self.nn.evaluate_board)

    def get_move(self, board):
        print "Guerilla is thinking..."
        return self.search.negamax(board)[1]