
from classifier import DigitClassifier
from perceptron import Perceptron
import random, utils, numpy
import matplotlib.pyplot as plt


class MultilayerPerceptronClassifier(DigitClassifier):
    """
        MultilayerPerceptronClassifier
    """
    def init(self):
        self.network = Network([784] + self.args.hidden_layers_sizes + [10], utils.sigmoid)

    def add_arguments(self, config):
        config.add_argument('-eta', default=3.0, type=float, help="Pas d'apprentissage")
        config.add_argument('-epochs', default=30, type=int, help="Nombre de cycles")
        config.add_argument('-batch-size', default=10, type=int, help="Taille d'un sous-échantillon")
        config.add_argument('-activation', '-act', default='sigmoid', choices=['sigmoid', 'tanh'], help="Fonction d'activation")
        config.add_argument('-hidden-layers-sizes', '-sizes', default=[20], type=int, nargs='+', help="Tailles des couches cachées")

    def train(self, images, labels):
        expected_outputs = [numpy.fromiter((labels[k] == i for i in range(10)), numpy.float64) for k in range(len(labels))]
        training_data = list(zip(images, expected_outputs))

        self.network.stochastic_gradient_descent(training_data, self.args.epochs, self.args.batch_size)
        self.plot_error()

    def plot_error(self):
        cost = self.network.cost
        numpy.save(self.args.outputs_folder + 'error.npy', numpy.array(cost))
        plt.plot(range(len(cost)), cost)
        plt.ylabel('erreur quadratique')
        plt.xlabel("cycle")
        plt.savefig(self.args.outputs_folder + 'error.png')

    def predict(self, image):
        y = self.network.feed_forward(image)
        return numpy.argmax(y)


class Network:
    """
        Network
    """
    def __init__(self, sizes, activation):
        self.sizes = sizes
        self.depth = len(sizes)-1
        self.layers = [Layer(sizes[k], sizes[k+1], activation) for k in range(self.depth)]

    def feed_forward(self, x):
        y = numpy.copy(x)
        for layer in self.layers:
            y = layer.output(y)
        return y


    def stochastic_gradient_descent(self, training_data, epochs=1000, batch_size=50, eta=0.1):
        self.cost = numpy.zeros(epochs)

        utils.timer_start()
        for epoch in range(epochs):
            mini_batches = self.get_mini_batches(training_data, batch_size)
            for mini_batch in mini_batches:
                nablaw, nablab = self.init_gradient()
                for x,t in mini_batch:
                    z = self.feed_forward(x)
                    nw, nb, cost = self.backpropagate(x, z, t)
                    self.cost[epoch] += cost

                    for d in range(self.depth):
                        nablaw[d] += nw[d]
                        nablab[d] += nb[d]

            # Modification des poids
            for d in range(self.depth):
                self.layers[d].W -= eta / batch_size * nablaw[d]
                self.layers[d].b -= eta / batch_size * nablab[d]


            self.cost[epoch] /= len(training_data)
            print('cycle', epoch, 'coût', self.cost[epoch])
            utils.print_remaining_time(epoch, epochs)


    def get_mini_batches(self, training_data, batch_size):
        random.shuffle(training_data)
        return [training_data[k:k+batch_size] for k in range(0, len(training_data), batch_size)]

    def init_gradient(self):
        nablaw = [numpy.zeros(self.layers[k].W.shape) for k in range(self.depth)]
        nablab = [numpy.zeros(self.layers[k].b.shape) for k in range(self.depth)]
        return nablaw, nablab

    def backpropagate(self, x, z, t):
        nablaw, nablab = self.init_gradient()
        delta = [numpy.zeros(nablab[d].shape, dtype=numpy.float64) for d in range(self.depth)]

        # Calcul sur la couche de sortie
        delta[-1] = numpy.multiply(z-t, utils.sigmoid_prime(self.layers[-1].w_inp))
        nablaw[-1] = numpy.outer(delta[-1], self.layers[-2].out)
        nablab[-1] = delta[-1]

        # Calcul sur les couches cachées
        for d in range(-2, -self.depth-1, -1):
            delta[d] = numpy.multiply(numpy.matmul(self.layers[d+1].W.T, delta[d+1]), utils.sigmoid_prime(self.layers[d].w_inp))
            if d == -self.depth:
                a = x
            else:
                a = self.layers[d-1].out
            nablaw[d] = numpy.outer(delta[d], a)
            nablab[d] = delta[d]

        cost = .5*sum([(z[k]-t[k])**2 for k in range(self.sizes[-1])])

        return nablaw, nablab, cost


class Layer:
    """
        Layer
    """
    def __init__(self, insize, outsize, activation):
        self.insize = insize
        self.outsize = outsize
        self.activation = activation
        self.W = numpy.random.randn(outsize, insize)
        self.b = numpy.random.randn(outsize)

    def output(self, x):
        self.w_inp = numpy.matmul(self.W, x)
        self.out = self.activation(self.w_inp)
        return self.out

    def set_weights(self, W, b):
        self.W = W
        self.b = b
