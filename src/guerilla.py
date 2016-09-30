import player
import neural_net
import search


class Guerilla(player.Player):
    def __init__(self, name, colour=None, _load_file=None):
        super(Guerilla, self).__init__(name, colour)
        self.nn = neural_net.NeuralNet(_load_file=_load_file)
        self.search = search.Search(self.nn.evaluate)

    def get_move(self, board):
        # print "Guerilla is thinking..."
        return self.search.run(board)[1]

        # TODO: Evaluate function abstraction. Places it appears: td_leaf and search.