from classifier import Classifier
from perceptron import Perceptron
import random
from utils import sigmoid, sigmoid_prime, print_remaining_time, timer_start
import numpy
import time
import matplotlib.pyplot as plt
import sys


class MultilayerPerceptronClassifier(Classifier):

    def init(self):
        self.network = Network(784, 20, 10)

    def train(self, images, labels):
        expected_outputs = [numpy.fromiter((labels[k] == i for i in range(10)), numpy.float64) for k in range(len(images))]
        self.network.backpropagate(images, expected_outputs, self.args.it)
        self.plot_error()

    def plot_error(self):
        cost = self.network.cost
        numpy.save('../output/error.npy', numpy.array(cost))
        plt.plot(range(len(cost)), cost)
        plt.ylabel('erreur quadratique')
        plt.xlabel("itération")
        plt.legend(loc='upper right')
        plt.savefig('../output/error.png')

    def predict(self, image):
        return numpy.argmax(self.network.feed_forward(image))

    def get_save(self):
        return [(self.network.layers[k].W, self.network.layers[k].b) for k in range(self.network.number_layers)]

    def load_save(self, save):
        k = 0
        for W,b in save:
            self.network.layers[k].set_weights(W, b)
            k += 1


class Network:

    def __init__(self, N_in, N_hidden, N_out):
        self.N_in = N_in
        self.N_hidden = N_hidden
        self.N_out = N_out
        self.layers = [Layer(N_in, N_hidden), Layer(N_hidden, N_out)]

    def feed_forward(self, x):
        y = numpy.copy(x)
        for layer in self.layers:
            y = layer.output(y)
        return y

    def train(self, inputs, expected_outputs, it=1000, eta=0.1):
        self.cost = []
        timer_start()
        for i in range(it):
            n = random.randint(0,len(inputs)-1)
            x, t = inputs[n], expected_outputs[n]
            z = self.feed_forward(x)

            nabla_h, nabla_s = self.backpropagate(x, z, t)

            self.layers[1].W -= eta * nabla_s
            self.layers[0].W -= eta * nabla_h
            print_remaining_time(i, it)
        print(self.cost)


    def backpropagate(self, x, z, t):
        nabla_h, nabla_s = numpy.zeros((self.N_hidden, self.N_in), dtype=numpy.float64), numpy.zeros((self.N_out, self.N_hidden), dtype=numpy.float64)

        delta_s = numpy.multiply((t-z)[:,None], sigmoid_prime(self.layers[1].w_inp)[None,:])
        nabla_s = numpy.dot(delta_s, self.layers[0].out[None,:])

        delta_h = numpy.multiply(numpy.matmul(self.layers[1].W.T, delta_s), sigmoid_prime(self.layers[0].w_inp))
        nabla_h = numpy.matmul(delta_h, x)

        self.cost.append(.5*sum([(z[k]-t[k])**2 for k in range(self.N_out)]))
        return nabla_h, nabla_s


'''
    def backpropagate(self, inputs, expected_outputs, it=10000, eta=0.1, mini_batch_size=100, plot_error=True):
        nb_examples = inputs.shape[0]
        timer_start()
        self.cost = []
        for i in range(it):
            deltak = numpy.zeros(self.Nout, dtype=numpy.float64)
            deltaj = numpy.zeros(self.Nhidden, dtype=numpy.float64)
            for k in range(mini_batch_size):
                n = random.randint(0, nb_examples-1)
                x, t = inputs[n], expected_outputs[n]
                z = self.feed_forward(x)

                for k in range(self.Nout):
                    deltak[k] +=  (t[k]-z[k])*sigmoid_prime(self.layers[1].inp[k])

                for j in range(self.Nhidden):
                    deltaj[j] += numpy.dot(deltak, self.layers[1].W[:,j])*sigmoid_prime(self.layers[0].inp[j])

            self.layers[1].W += eta/mini_batch_size*numpy.array([deltak[k]*self.layers[0].out for k in range(self.Nout)], dtype=numpy.float64)
            self.layers[1].b += eta/mini_batch_size*deltak
            self.layers[0].W += eta/mini_batch_size*numpy.array([deltaj[j]*x for j in range(self.Nhidden)], dtype=numpy.float64)
            self.layers[0].b += eta/mini_batch_size*deltaj

            self.cost.append(.5*sum([(t[k] - z[k])**2 for k in range(self.Nout)]))
            print_remaining_time(i, it)
'''


class Layer:

    def __init__(self, insize, outsize, activation=sigmoid):
        self.insize = insize
        self.outsize = outsize
        self.W = numpy.random.rand(outsize, insize+1)
        self.b = numpy.random.rand(outsize)
        self.activation = activation

    def output(self, x):
        self.w_inp = numpy.matmul(self.W, numpy.append(x, [1]))
        self.out = self.activation(self.w_inp)
        return self.out

    def set_weights(self, W, b):
        self.W = W
        self.b = b
