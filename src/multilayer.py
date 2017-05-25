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
        self.network.train(images, expected_outputs, self.args.it)
        self.plot_error()

    def plot_error(self):
        cost = self.network.cost
        numpy.save('../output/error.npy', numpy.array(cost))
        plt.plot(range(len(cost)), cost)
        plt.ylabel('erreur quadratique')
        plt.xlabel("itération")
        plt.savefig('../output/error.png')

    def predict(self, image):
        return numpy.argmax(self.network.feed_forward(image)[0])

    def get_save(self):
        return [(self.network.layers[k].W, self.network.layers[k].b) for k in range(self.network.number_layers)]

    def load_save(self, save):
        k = 0
        for W, b in save:
            self.network.layers[k].set_weights(W)
            k += 1


class Network:

    def __init__(self, N_in, N_hidden, N_out):
        self.N_in = N_in
        self.N_hidden = N_hidden
        self.N_out = N_out
        self.layers = [Layer(N_in, N_hidden), Layer(N_hidden, N_out)]
        self.number_layers = len(self.layers)
        print(self.layers[0].W.shape)
        print(self.layers[1].W.shape)

    def feed_forward(self, x):
        y = numpy.copy(x).reshape((self.N_in, 1))
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

            nabla_hw, nabla_hb, nabla_sw, nabla_sb = self.backpropagate(x, z, t)

            self.layers[1].W -= eta * nabla_sw
            self.layers[1].b -= eta * nabla_sb

            self.layers[0].W -= eta * nabla_hw
            self.layers[0].b -= eta * nabla_hb

            print_remaining_time(i, it)

            if i % (it//100) == 0:
                print('coût : ', self.cost[len(self.cost)-1])


    def backpropagate(self, x, z, t):
        print(z, t, z-t)
        delta_s = numpy.multiply((z-t), sigmoid_prime(self.layers[1].w_inp)).reshape((1, self.N_out))
        nabla_sw = numpy.matmul(delta_s, self.layers[0].out.reshape((1,self.layers[0].outsize)))
        nabla_sb = delta_s

        delta_h = numpy.multiply(numpy.matmul(delta_s.T, self.layers[1].W), sigmoid_prime(self.layers[0].w_inp).reshape((self.N_hidden, 1)))
        nabla_hw = numpy.matmul(delta_h, x.reshape((1, self.layers[0].insize)))
        nabla_hb = delta_h

        self.cost.append(.5*sum([(z[0][k]-t[k])**2 for k in range(self.N_out)]))
        return nabla_hw, nabla_hb, nabla_sw, nabla_sb

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
        self.W = numpy.random.rand(outsize, insize)
        self.b = numpy.random.rand(outsize, 1)
        self.activation = activation

    def output(self, x):
        self.w_inp = numpy.matmul(self.W, x) + self.b
        self.out = self.activation(self.w_inp)
        return self.out

    def set_weights(self, W, b):
        self.W = W
        self.b = b
