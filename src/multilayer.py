
from classifier import DigitClassifier
from perceptron import Perceptron
import random, utils, numpy
import matplotlib.pyplot as plt


class MultilayerPerceptronClassifier(DigitClassifier):
    """
        MultilayerPerceptronClassifier
    """
    def init(self):
        self.network = Network([784, 20, 10], self.args.activation)

    def add_arguments(self, config):
        config.add_argument('--eta', default=0.01, type=float, help="Pas d'apprentissage")
        config.add_argument('--it', default=1000, type=int, help="Nombre d'exemple d'apprentissage")
        config.add_argument('--activation', default='sgn', choices=['sigmoid', 'tanh'], help="Fonction d'activation")

    def train(self, images, labels):
        if self.args.activation == 'tanh':
            expected_outputs = [numpy.fromiter((1 if labels[k] == i else -1 for i in range(10)), numpy.float64) for k in range(len(images))]
        else:
            expected_outputs = [numpy.fromiter((labels[k] == i for i in range(10)), numpy.float64) for k in range(len(images))]

        self.network.mini_batch_train(images, expected_outputs, self.args.it, numpy.float64(self.args.eta), verbose=True)
        self.plot_error()

    def plot_error(self):
        cost = self.network.cost
        numpy.save(self.args.outputs_folder + 'error.npy', numpy.array(cost))
        plt.plot(range(len(cost)), cost)
        plt.ylabel('erreur quadratique')
        plt.xlabel("cycle")
        #plt.xscale('log')
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
        'ntanh': utils.ntanh,
        'nsig': lambda x: 1./(1+numpy.exp(-x))
    }

    df = {
        'sigmoid': utils.sigmoid_prime,
        'tanh' : utils.tanh_prime,
        'ntanh' : utils.ntanh_prime,
        'nsig' : lambda x: self.f['nsig'](x)*(1-self.f['nsig'](x))
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
            #if i >= 2000:
            #    eta = 0.01*int(((0.001-0.1)/(20000-2000)*(i-2000)+0.1)/0.01)
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

    def mini_batch_train(self, inputs, expected_outputs, it=1000, eta=0.1, verbose=False, mini_batch_size=50):
        epochs = it // mini_batch_size
        self.cost = numpy.zeros(epochs)

        nablaw, nablab = self.init_gradient()

        utils.timer_start()
        for i in range(epochs):
            self.costk = 0.
            for k in range(mini_batch_size):
                n = random.randint(0,len(inputs)-1)
                x, t = inputs[n], expected_outputs[n]
                z = self.feed_forward(x)

                nw, nb = self.backpropagate(x, z, t)

                for d in range(self.depth):
                    nablaw[d] += nw[d]
                    nablab[d] += nb[d]

            self.cost[i] = self.costk / mini_batch_size

            for d in range(self.depth):
                self.layers[d].W -= eta / mini_batch_size * nablaw[d]
                self.layers[d].b -= eta / mini_batch_size * nablab[d]

            nablaw, nablab = self.init_gradient()

            if verbose and i % (epochs // 100) == 0:
                print('cycle', i, 'coût', self.cost[i])
                utils.print_remaining_time(i, epochs)

    def momentum_mini_batch_train(self, inputs, expected_outputs, it=1000, eta=0.1, verbose=False, mini_batch_size=50):
        epochs = it // mini_batch_size
        alpha = .01
        self.cost = numpy.zeros(epochs)

        nablaw, nablab = self.init_gradient()
        prev_nablaw, prev_nablab = self.init_gradient()

        utils.timer_start()
        for i in range(epochs):
            self.costk = 0.
            for k in range(mini_batch_size):
                n = random.randint(0,len(inputs)-1)
                x, t = inputs[n], expected_outputs[n]
                z = self.feed_forward(x)

                nw, nb = self.backpropagate(x, z, t)

                for d in range(self.depth):
                    nablaw[d] += nw[d]
                    nablab[d] += nb[d]

            self.cost[i] = self.costk / mini_batch_size

            for d in range(self.depth):
                nablaw[d] /= mini_batch_size
                nablab[d] /= mini_batch_size
                self.layers[d].W -= (1-alpha) * eta * nablaw[d] + alpha * eta * prev_nablaw[d]
                self.layers[d].b -= (1-alpha) * eta * nablab[d] + alpha * eta * prev_nablab[d]

            prev_nablaw, prev_nablab = nablaw, nablab

            nablaw, nablab = self.init_gradient()

            if verbose and i % (epochs // 100) == 0:
                print('cycle', i, 'coût', self.cost[i])
                utils.print_remaining_time(i, epochs)

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

        self.costk += .5*sum([(z[k]-t[k])**2 for k in range(self.sizes[-1])])

        return nablaw, nablab


class Layer:
    """
        Leyer
    """
    def __init__(self, insize, outsize, activation=utils.sigmoid):
        self.insize = insize
        self.outsize = outsize
        self.W = numpy.random.uniform(-1, 1, (outsize, insize))
        self.b = numpy.random.uniform(-1, 1, outsize)
        self.activation = activation

    def output(self, x):
        self.w_inp = numpy.matmul(self.W, x) + self.b
        self.out = self.activation(self.w_inp)
        return self.out

    def set_weights(self, W, b):
        self.W = W
        self.b = b
