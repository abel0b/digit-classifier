
from classifier import Classifier
from perceptron import Perceptron
import random, utils, numpy
import matplotlib.pyplot as plt


class MultilayerPerceptronClassifier(Classifier):

    def init(self):
        self.network = Network(784, 20, 10, activation=self.args.activation)

    def train(self, images, labels):
        if self.args.activation == 'tanh':
            expected_outputs = [numpy.fromiter((1 if labels[k] == i else -1 for i in range(10)), numpy.float64) for k in range(len(images))]
        else:
            expected_outputs = [numpy.fromiter((labels[k] == i for i in range(10)), numpy.float64) for k in range(len(images))]

        self.network.mini_batch_train(images, expected_outputs, self.args.it, numpy.float64(self.args.eta), verbose=True)
        self.plot_error()

    def plot_error(self):
        cost = self.network.cost
        numpy.save(self.args.error_file, numpy.array(cost))
        plt.plot(range(len(cost)), cost)
        plt.ylabel('erreur quadratique')
        plt.xlabel("cycle")
        plt.xscale('log')
        plt.savefig(self.args.error_img)

    def predict(self, image):
        y = self.network.feed_forward(image)
        return numpy.argmax(y)

    def get_save(self):
        return [(self.network.layers[k].W, self.network.layers[k].b) for k in range(self.network.number_layers)]

    def load_save(self, save):
        k = 0
        for W, b in save:
            self.network.layers[k].set_weights(W, b)
            k += 1


class Network:

    f = {
        'sigmoid': utils.sigmoid,
        'tanh': utils.tanh,
        'ntanh': utils.ntanh,
    }

    df = {
        'sigmoid': utils.sigmoid_prime,
        'tanh' : utils.tanh_prime,
        'ntanh' : utils.ntanh_prime
    }

    def __init__(self, N_in, N_hidden, N_out, activation):
        self.N_in = N_in
        self.N_hidden = N_hidden
        self.N_out = N_out
        self.layers = [Layer(N_in, N_hidden, self.f[activation]), Layer(N_hidden, N_out, self.f[activation])]
        self.sigmoid = [self.f[activation], self.f[activation]]
        self.sigmoid_prime = [self.df[activation], self.df[activation]]
        self.number_layers = len(self.layers)

    def feed_forward(self, x):
        y = numpy.copy(x)
        for layer in self.layers:
            y = layer.output(y)
        return y

    def train(self, inputs, expected_outputs, it=1000, eta=0.1, verbose=False):
        self.cost = []
        utils.timer_start()
        for i in range(it):
            #if i >= 2000:
            #    eta = 0.01*int(((0.001-0.1)/(20000-2000)*(i-2000)+0.1)/0.01)
            n = random.randint(0,len(inputs)-1)
            x, t = inputs[n], expected_outputs[n]
            z = self.feed_forward(x)

            nabla_hw, nabla_hb, nabla_sw, nabla_sb = self.backpropagate(x, z, t)

            self.layers[1].W -= eta * nabla_sw
            self.layers[1].b -= eta * nabla_sb

            self.layers[0].W -= eta * nabla_hw
            self.layers[0].b -= eta * nabla_hb

            if verbose and i % (it//100) == 0:
                utils.print_remaining_time(i, it)
                print('coût', self.cost[len(self.cost)-1])

    def mini_batch_train(self, inputs, expected_outputs, it=1000, eta=0.1, verbose=False):
        mini_batch_size = 50
        epochs = it // mini_batch_size
        self.cost = numpy.zeros(epochs)
        nabla_hw, nabla_hb, nabla_sw, nabla_sb = numpy.zeros(self.layers[0].W.shape), numpy.zeros(self.layers[0].b.shape), numpy.zeros(self.layers[1].W.shape), numpy.zeros(self.layers[1].b.shape)
        utils.timer_start()
        for i in range(epochs):
            self.costk = 0.
            for k in range(mini_batch_size):
                n = random.randint(0,len(inputs)-1)
                x, t = inputs[n], expected_outputs[n]
                z = self.feed_forward(x)

                nhw, nhb, nsw, nsb, costk = self.backpropagate(x, z, t)

                nabla_sw += nsw
                nabla_sb += nsb
                nabla_hw += nhw
                nabla_hb += nhb
            self.cost[i] = costk / mini_batch_size
            '''print(t, z)
            for i in range(28):
                msg = ''
                for j in range(28):
                    if inputs[n][i*28+j] > 0.7:
                        msg += '▮'
                    else:
                        msg += ' '
                print(msg)'''
            self.layers[1].W -= eta / mini_batch_size * nabla_sw
            self.layers[1].b -= eta / mini_batch_size * nabla_sb

            self.layers[0].W -= eta / mini_batch_size * nabla_hw
            self.layers[0].b -= eta / mini_batch_size * nabla_hb
            nabla_hw, nabla_hb, nabla_sw, nabla_sb = numpy.zeros(self.layers[0].W.shape), numpy.zeros(self.layers[0].b.shape), numpy.zeros(self.layers[1].W.shape), numpy.zeros(self.layers[1].b.shape)

            if verbose and i % (epochs // 100) == 0:
                print('cycle', i, 'coût', self.cost[i])
                utils.print_remaining_time(i, epochs)


    def backpropagate(self, x, z, t):
        delta_s = numpy.multiply(z-t, self.sigmoid_prime[1](self.layers[1].w_inp))
        nabla_sw = numpy.outer(delta_s, self.layers[0].out)
        nabla_sb = delta_s

        delta_h = numpy.multiply(numpy.matmul(delta_s, self.layers[1].W), self.sigmoid_prime[0](self.layers[0].w_inp))
        nabla_hw = numpy.outer(delta_h, x)
        nabla_hb = delta_h

        self.costk += .5*sum([(z[k]-t[k])**2 for k in range(self.N_out)])

        return nabla_hw, nabla_hb, nabla_sw, nabla_sb, self.costk


class Layer:

    def __init__(self, insize, outsize, activation=utils.sigmoid):
        self.insize = insize
        self.outsize = outsize
        if activation == utils.tanh:
            self.W = numpy.random.uniform(-1, 1, (outsize, insize))
            self.b = numpy.random.uniform(-1, 1, outsize)
        else:
            self.W = numpy.random.rand(outsize, insize)
            self.b = numpy.random.rand(outsize)
        self.activation = activation

    def output(self, x):
        self.w_inp = numpy.matmul(self.W, x) + self.b
        self.out = self.activation(self.w_inp)
        return self.out

    def set_weights(self, W, b):
        self.W = W
        self.b = b
