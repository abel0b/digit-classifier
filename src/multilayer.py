from classifier import Classifier
from numpy import array, argmax, zeros
from perceptron import Perceptron
from random import randint
from utils import activation_prime, sectostr
from time import time

class MultilayerPerceptronClassifier(Classifier):

    def __init__(self):
        self.network = Network(784,15,10)

    def train(self, images, labels, start=0):
        expected_outputs = [[float(labels[k]==i) for i in range(10)] for k in range(len(images))]
        self.network.backpropagate(images, expected_outputs, start)

    def predict(self, image):
        return argmax(self.network.output(image))


class Network:

    def __init__(self, Nin, Nhidden, Nout):
        sizes = [Nin, Nhidden, Nout]
        self.number_layers = len(sizes)-1
        self.Nin = Nin
        self.Nhidden = Nhidden
        self.Nout = Nout
        self.input_size = sizes[0]
        self.layers = [Layer(sizes[layer],sizes[layer+1]) for layer in range(self.number_layers)]

    def output(self, x):
        next_input = x
        for k in range(self.number_layers):
            next_input = self.layers[k].output(next_input)
        return next_input

    def backpropagate(self, inputs, expected_outputs, start, it=1000000, eta=0.1):
        examples = []
        ti = 0
        for i in range(it):
            n = randint(0,len(inputs)-1)
            examples.append(n)
            x = inputs[n]
            t = expected_outputs[n]
            z = self.output(x)
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
                print(str(ti)+'% :' + sectostr((time()-start)/i/it*(it-i)) + ' remaining')

class Layer:
    size = 0

    def __init__(self, insize, outsize):
        self.insize = insize
        self.outsize = outsize
        self.perceptrons = [Perceptron(insize) for k in range(outsize)]

    def output(self, input):
        self.out = array([self.perceptrons[k].output(input) for k in range(self.outsize)], dtype="float64")
        return self.out
