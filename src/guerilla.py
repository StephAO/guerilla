import player
import neural_net
import search


class Guerilla(player.Player):

    def __init__(self, name, colour=None, _load_file=None):
        super(Guerilla, self).__init__(name, colour)
        self.nn = neural_net.NeuralNet(load_file=_load_file)
        self.search = search.Search(self.nn.evaluate)

    def __enter__(self):
        self.nn.start_session()
        self.nn.init_graph()
        return self

    def __exit__(self, type, value, traceback):
        if type is not None:
            print type, value, traceback
        self.nn.close_session()

    def get_move(self, board):
        # print "Guerilla is thinking..."
        return self.search.run(board)[1]

if __name__ == '__main__':
    print "test"
    with Guerilla('test','w') as g:
        print g.nn.get_weights(g.nn.all_weights)[0][0][0][0]
