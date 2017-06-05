from classifier import DigitClassifier
import random
import numpy
import time
import matplotlib.pyplot as plt

from utils import sgn, sigmoid, timer_start, print_remaining_time


class PerceptronClassifier(DigitClassifier):
    activation = {
        'sgn': sgn,
        'sigmoid': sigmoid
    }

    def init(self):
        def sgn(x): return 1 if x >= 0 else 0
        self.perceptrons = [Perceptron(28 * 28, activation_function=self.activation[self.args.activation]) for i in range(10)]

    def add_arguments(self, config):
        config.add_argument('--eta', default=0.01, type=float, help="Pas d'apprentissage")
        config.add_argument('--it', default=1000, type=int, help="Nombre d'exemple d'apprentissage")
        config.add_argument('--activation', default='sgn', choices=['sgn', 'sigmoid'], help="Fonction d'activation")

    def train(self, images, labels):
        timer_start()
        for i in range(10):
            self.perceptrons[i].train(images, labels, i, it=self.args.it, eta=self.args.eta)
            print_remaining_time(i, 10)
        self.plot_error(100)

    def predict(self, image):
        output = numpy.array([self.perceptrons[i].output(image) for i in range(10)])
        if self.args.activation == 'sgn':
            return self.predict_sgn(output)
        else:
            return self.predict_sigmoid(output)

    def predict_sigmoid(self, output):
        return numpy.argmax(output)

    def predict_sgn(self, output):
        if numpy.nonzero(output)[0].size == 1:
            return numpy.argmax(output)
        else:
            return -1

    def plot_error(self, it):
        for i in range(10):
            plt.plot(range(it), self.perceptrons[i].error, label=i)
            # plt.xscale('log')
        plt.ylabel('erreur quadratique')
        plt.xlabel("itération")
        plt.legend(loc='upper right')
        plt.savefig(self.args.outputs_folder + 'error.png')
        numpy.save(self.args.outputs_folder + 'error.npy', numpy.array([self.perceptrons[i].error for i in range(10)]))


class Perceptron:
    def __init__(self, d, activation_function=sgn):
        self.d = d
        self.activation_function = activation_function
        self.bias = random.random()
        self.weights = numpy.array([random.random() for i in range(d)])
        self.error = []

    def output(self, x):
        self.weighted_input = self.bias + numpy.vdot(x, self.weights)
        r = self.activation_function(self.weighted_input)
        return r

    def update_weights(self, delta, vec):
        self.weights += delta * numpy.array(vec, dtype="float64")
        self.bias += delta

    def train(self, images, labels, digit, it=10000, plot_error=False, eta=0.01):
        errnum = 100
        self.error = numpy.zeros(errnum)
        imlist = [random.randint(0, len(images) - 1) for k in range(it)]
        for k in range(it):
            n = imlist[k]
            imlist.append(n)
            x = numpy.array(images[n], dtype="float64")
            output = self.output(x)
            y = int(digit == labels[n])
            error = y - output
            if output != y:
                self.weights += error * eta * x
                self.bias += error * eta
            if plot_error and k < errnum:
                for i in range(errnum):
                    self.error[k] += (float(digit == labels[imlist[i]]) -
                                      self.output(numpy.array(images[imlist[i]], dtype="float64")))**2
                self.error[k] /= errnum

    def success_rate(self, data):
        s = 0
        for x, y in data:
            if self.eval(x) == y:
                s += 1
        return s / len(data) * 100
