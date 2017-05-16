from classifier import Classifier
from perceptron import Perceptron
from random import randint
from utils import activation_prime, sectostr
import numpy, time
import matplotlib.pyplot as plt

class MultilayerPerceptronClassifier(Classifier):

    def __init__(self):
        self.network = Network(784,15,10)

    def train(self, images, labels, start=0):
        expected_outputs = [[float(labels[k]==i) for i in range(10)] for k in range(len(images))]
        self.network.backpropagate(images, expected_outputs, start, self.cfg['train'])
        if self.cfg['plot_error']:
            self.plot_error()

    def plot_error(self):
        cost = self.network.cost
        numpy.save('../output/error.npy', numpy.array(cost))
        l = numpy.array(list(range(len(cost))))
        plt.plot([100*k for k in range(len(l))], cost, label='J')
        plt.ylabel('erreur quadratique')
        plt.xlabel("n° échantillon")
        plt.legend(loc='upper right')
        plt.savefig('../output/error.png')

    def predict(self, image):
        return numpy.argmax(self.network.output(image))

    def get_save(self):
        return [[(self.network.layers[k].perceptrons[i].weights, self.network.layers[k].perceptrons[i].bias) for i in range(self.network.sizes[k+1])] for k in range(self.network.number_layers)]

    def load_save(self, save):
        for k in range(self.network.number_layers):
            self.network.layers[k].set_weights(save[k])


class Network:

    def __init__(self, Nin, Nhidden, Nout):
        self.sizes = [Nin, Nhidden, Nout]
        self.number_layers = len(self.sizes)-1
        self.Nin = Nin
        self.Nhidden = Nhidden
        self.Nout = Nout
        self.input_size = self.sizes[0]
        self.layers = [Layer(self.sizes[layer],self.sizes[layer+1]) for layer in range(self.number_layers)]

    def output(self, x):
        next_input = x
        for k in range(self.number_layers):
            next_input = self.layers[k].output(next_input)
        return next_input

    def backpropagate(self, inputs, expected_outputs, start, it=1000, eta=0.1, subsample_size=100, plot_error=False):
        examples = []
        ti = 0
        self.cost = [0 for i in range(it//subsample_size)]
        sample_no = 0
        for i in range(it):
            if i == 20000:
                eta = 0.01
            elif i == 90000:
                eta = 0.001
            n = randint(0,len(inputs)-1)
            examples.append(n)
            x = inputs[n]
            t = expected_outputs[n]
            z = self.output(x)


            if i > 0 and i % subsample_size == 0:
                sample_no += 1
            self.cost[sample_no] += 1/2*sum([(z[k] - t[k])**2 for k in range(self.Nout)])


            # Mise à jour des poids de la couche de sortie
            for k in range(self.Nout):
                deltak = eta*(t[k]-z[k])*activation_prime(self.layers[1].perceptrons[k].weighted_input)
                self.layers[1].perceptrons[k].update_weights(eta*deltak,self.layers[0].out)

            # Mise à jour des poids de la couche d'entrée
            for j in range(self.Nhidden):
                deltaj = 0.
                for k in range(self.Nout):
                    deltak = eta*(t[k]-z[k])*activation_prime(self.layers[1].perceptrons[k].weighted_input)
                    deltaj += deltak*self.layers[1].perceptrons[k].weights[j]
                deltaj *= activation_prime(self.layers[0].perceptrons[j].weighted_input)
                self.layers[0].perceptrons[j].update_weights(deltaj, x)

            if i > 0 and i % (it//100) == 0:
                ti += 1
                print(str(ti)+'% : ' + sectostr((it-i)*(time.time()-start)/i) + ' remaining')

class Layer:
    size = 0

    def __init__(self, insize, outsize):
        self.insize = insize
        self.outsize = outsize
        self.perceptrons = [Perceptron(insize) for k in range(outsize)]

    def output(self, input):
        self.out = numpy.array([self.perceptrons[k].output(input) for k in range(self.outsize)], dtype="float64")
        return self.out

    def set_weights(self, weights):
        i = 0
        for omega, bias in weights:
            self.perceptrons[i].weights = omega
            self.perceptrons[i].bias = bias
            i += 1
