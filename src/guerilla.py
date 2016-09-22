import player
import neural_net
import search


class Guerilla(player.Player):
    def __init__(self, name, colour=None, load_file=None):
        super(Guerilla, self).__init__(name, colour)
        self.nn = neural_net.NeuralNet(load_weights=(load_file is not None), load_file=load_file)
        self.search = search.Search(self.nn.evaluate)

    def get_move(self, board):
        # print "Guerilla is thinking..."
        return self.search.run(board)[1]

        # TODO: Evaluate function abstraction. Places it appears: td_leaf and search.