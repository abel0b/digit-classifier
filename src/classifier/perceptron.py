from classifier.classifier import Classifier
from random import randint
from numpy import vdot, array, tanh, zeros
from random import randint
import matplotlib.pyplot as plt

class PerceptronClassifier(Classifier):
    def __init__(self):
        self.perceptrons = [Perceptron(28*28) for i in range(10)]

    def train(self, images, labels, it=10000):
        for i in range(10):
            self.perceptrons[i].train(images,labels,i,it)
        self.plotError(1000)

    def recognize(self,image):
        R = [0 for i in range(10)]
        for i in range(10):
            if self.perceptrons[i].output(image) == 1:
                R[i] = 1
        if R.count(1) == 1:
            return R.index(1)
        else:
            return "_"

    def plotError(self, it):
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
    def __init__(self, d, bias=0):
        self.d = d
        #self.activation_function = lambda t: 0 if t < 0 else 1
        self.activation_function = tanh
        self.bias = bias
        self.weights = array([0. for i in range(d)])
        self.error = []

    def output(self,x):
        r = self.activation_function(self.bias + vdot(x,self.weights))
        return 1 if r>=0.5 else 0

    def train(self,images, labels,digit,it=10000, eta=0.01):
        errnum = 1000
        self.error = zeros(errnum)
        imlist = [randint(0,len(images)-1) for k in range(it)]
        for k in range(it):
            n = imlist[k]
            imlist.append(n)
            x = array(images[n],dtype="float64")
            output = self.output(x)
            y = int(digit==labels[n])
            error = y-output
            if output != y:
                self.weights += error*eta*x
                self.bias += error*eta
            if k < errnum:
                for i in range(errnum):
                    self.error[k] += (float(digit==labels[imlist[i]])-self.output(array(images[imlist[i]],dtype="float64")))**2
                self.error[k] /= errnum
                print(k)

    def success_rate(self,data):
        s = 0
        for x,y in data:
            if self.eval(x) == y:
                s += 1
        return s/len(data)*100
