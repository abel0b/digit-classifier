
from classifier import DigitClassifier
from perceptron import Perceptron
import random, utils, numpy
import matplotlib.pyplot as plt


class MultilayerPerceptronClassifier(DigitClassifier):
    """
        MultilayerPerceptronClassifier
    """
    def init(self):
        self.network = Network([784] + self.args.hidden_layers_sizes + [10], self.args.activation)

    def add_arguments(self, config):
        config.add_argument('-eta', default=0.01, type=float, help="Pas d'apprentissage")
        config.add_argument('-epochs', default=1000, type=int, help="Nombre de cycles")
        config.add_argument('-batch-size', default=100, type=int, help="Taille d'un sous-échantillon")
        config.add_argument('-activation', '-act', default='tanh', choices=['sigmoid', 'tanh'], help="Fonction d'activation")
        config.add_argument('-hidden-layers-sizes', '-sizes', default=[20], type=int, nargs='+', help="Tailles des couches cachées")

    def train(self, images, labels):
        expected_outputs = [numpy.fromiter((labels[k] == i for i in range(10)), numpy.float64) for k in range(len(labels))]
        training_data = list(zip(images, expected_outputs))

        self.network.stochastic_gradient_descent(training_data, self.args.epochs, self.args.batch_size, verbose=True)
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
    f = {
        'sigmoid': utils.sigmoid,
        'tanh': utils.tanh,
        'softplus' : utils.softplus
    }

    df = {
        'sigmoid': utils.sigmoid_prime,
        'tanh' : utils.tanh_prime,
        'softplus' : utils.sigmoid
    }

    def __init__(self, sizes, activation='sigmoid'):
        self.sizes = sizes
        self.depth = len(sizes)-1
        self.layers = [Layer(sizes[k], sizes[k+1], self.f[activation]) for k in range(self.depth)]
        self.sigmoid = [self.f[activation] for k in range(self.depth)]
        self.sigmoid_prime = [self.df[activation] for k in range(self.depth)]

    def feed_forward(self, x):
        y = numpy.copy(x)
        for layer in self.layers:
            y = layer.output(y)
        return y

    def train(self, inputs, expected_outputs, it=1000, eta=0.1, verbose=False):
        self.cost = numpy.zeros(it)
        utils.timer_start()
        for i in range(it):
            n = random.randint(0,len(inputs)-1)
            x, t = inputs[n], expected_outputs[n]
            z = self.feed_forward(x)

            self.costk = 0.
            nablaw, nablab = self.backpropagate(x, z, t)
            self.cost[i] = self.costk

            for d in range(self.depth):
                self.layers[d].W -= eta * nablaw[d]
                self.layers[d].b -= eta * nablab[d]

            if verbose and i % (it//100) == 0:
                utils.print_remaining_time(i, it)
                print('coût', self.costk)

    def stochastic_gradient_descent(self, training_data, epochs=1000, batch_size=50, eta=0.1, verbose=False):
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
        delta[-1] = numpy.multiply(z-t, self.sigmoid_prime[-1](self.layers[-1].w_inp + self.layers[-1].b))
        nablaw[-1] = numpy.outer(delta[-1], self.layers[-2].out)
        nablab[-1] = delta[-1]

        # Calcul sur les couches cachées
        for d in range(-2, -self.depth-1, -1):
            delta[d] = numpy.multiply(numpy.matmul(delta[d+1], self.layers[d+1].W), self.sigmoid_prime[d](self.layers[d].w_inp + self.layers[d].b))
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
    def __init__(self, insize, outsize, activation=utils.sigmoid, use_bias=True):
        self.use_bias = use_bias
        self.insize = insize
        self.outsize = outsize
        self.W = numpy.random.uniform(-1, 1, (outsize, insize))
        self.b = numpy.random.uniform(-1, 1, outsize)
        self.activation = activation

    def output(self, x):
        if self.use_bias:
            self.w_inp = numpy.matmul(self.W, x) + self.b
        else:
            self.w_inp = numpy.matmul(self.W, x)
        self.out = self.activation(self.w_inp)
        return self.out

    def set_weights(self, W, b):
        self.W = W
        self.b = b
