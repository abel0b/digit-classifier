from classifier import Classifier
import random, numpy, time
import matplotlib.pyplot as plt

from utils import sigmoid, timer_start, print_remaining_time

class PerceptronClassifier(Classifier):
    def init(self):
        sgn = lambda x: 1 if x >= 0 else 0
        self.perceptrons = [Perceptron(28*28, activation_function = sgn) for i in range(10)]

    def train(self, images, labels):
        timer_start()
        for i in range(10):
            self.perceptrons[i].train(images,labels,i, it=self.args.it,eta=self.args.eta )
            print_remaining_time(i,10)
        self.plot_error(100)

    def predict(self,image):
        R = numpy.array([self.perceptrons[i].output(image) for i in range(10)])
        return numpy.argmax(R)

    def get_save(self):
        return [(self.perceptrons[i].weights, self.perceptrons[i].bias) for i in range(10)]

    def load_save(self, save):
        i = 0
        for weights, bias in save:
            self.perceptrons[i].weights = weights
            self.perceptrons[i].bias = bias
            i += 1

    def plot_error(self, it):
        for i in range(10):
            plt.plot(range(it),self.perceptrons[i].error,label=i)
            #plt.xscale('log')
        plt.ylabel('erreur quadratique')
        plt.xlabel("itération")
        plt.legend(loc='upper right')
        plt.savefig('../output/error.png')

    def close(self):
        plt.close()


class Perceptron:
    def __init__(self, d, bias=0, activation_function = sigmoid):
        self.d = d
        self.activation_function = activation_function
        self.bias = bias
        self.weights = numpy.array([random.random() for i in range(d)])
        self.error = []

    def output(self,x):
        self.weighted_input = self.bias + numpy.vdot(x,self.weights)
        r = self.activation_function(self.weighted_input)
        return r

    def update_weights(self, delta, vec):
        self.weights += delta*numpy.array(vec, dtype="float64")
        self.bias += delta

    def train(self,images, labels,digit,it=10000, plot_error = False, eta=0.01):
        errnum = 100
        self.error = numpy.zeros(errnum)
        imlist = [random.randint(0,len(images)-1) for k in range(it)]
        for k in range(it):
            n = imlist[k]
            imlist.append(n)
            x = numpy.array(images[n],dtype="float64")
            output = self.output(x)
            y = int(digit==labels[n])
            error = y-output
            if output != y:
                self.weights += error*eta*x
                self.bias += error*eta
            if plot_error and k < errnum:
                for i in range(errnum):
                    self.error[k] += (float(digit==labels[imlist[i]])-self.output(numpy.array(images[imlist[i]],dtype="float64")))**2
                self.error[k] /= errnum

    def success_rate(self,data):
        s = 0
        for x,y in data:
            if self.eval(x) == y:
                s += 1
        return s/len(data)*100
