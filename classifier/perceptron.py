from classifier.classifier import Classifier
from random import randint
from numpy import vdot,array,tanh
from random import randint

class PerceptronClassifier(Classifier):
    def __init__(self):
        self.perceptrons = [Perceptron(28*28) for i in range(10)]

    def train(self, images, labels, it=1000):
        for i in range(10):
            self.perceptrons[i].train(images,labels,i)

    def recognize(self,image):
        R = [0 for i in range(10)]
        for i in range(10):
            if self.perceptrons[i].output(image) == 1:
                R[i] = 1
        if R.count(1) == 1:
            return R.index(1)
        else:
            return "_"

class Perceptron:
    def __init__(self, d, bias=0):
        self.d = d
        #self.activation_function = lambda t: 0 if t < 0 else 1
        self.activation_function = tanh
        self.bias = bias
        self.weights = array([0. for i in range(d)])

    def output(self,x):
        r = self.activation_function(self.bias + vdot(x,self.weights))
        return 1 if r>=0.5 else 0

    def train(self,images, labels,digit,it=10000, eta=0.01):
        for k in range(it):
            n = randint(0,len(images)-1)
            x = array(images[n],dtype="float64")
            output = self.output(x)
            y = int(digit==labels[n])
            if output != y:
                self.weights += (y-output)*eta*x
                self.bias += (y-output)*eta

    def success_rate(self,data):
        s = 0
        for x,y in data:
            if self.eval(x) == y:
                s += 1
        return s/len(data)*100
