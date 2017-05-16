from classifier import Classifier
import random, numpy, time
import matplotlib.pyplot as plt

class PerceptronClassifier(Classifier):
    def __init__(self):
        sgn = lambda x: 1 if x >= 0 else 0
        self.perceptrons = [Perceptron(28*28, activation_function = sgn) for i in range(10)]

    def train(self, images, labels, start=0, plot_error=False):
        it = self.cfg['train']
        for i in range(10):
            self.perceptrons[i].train(images,labels,i,self.cfg['train'],self.cfg['plot_error'])
            if i < 9:
                print(str(i+1) + '/10 : ' + str(int((10-i-1)*(time.time()-start)/(i+1))) + 's remaining')
        if self.cfg['plot_error']:
            self.plot_error(100)

    def get_save(self):
        return [(self.perceptrons[i].weights, self.perceptrons[i].bias) for i in range(10)]

    def load_save(self, save):
        i = 0
        for weights, bias in save:
            self.perceptrons[i].weights = weights
            self.perceptrons[i].bias = bias
            i += 1

    def predict(self,image):
        R = [self.perceptrons[i].output(image) for i in range(10)]
        if R.count(1) == 1:
            return R.index(1)
        else:
            return '_'

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
    def __init__(self, d, bias=0, activation_function = numpy.tanh):
        self.d = d
        #self.activation_function = lambda t: 0 if t < 0 else 1
        self.activation_function = activation_function
        self.activation_prime = lambda x: 1. + numpy.tanh(x)**2
        self.bias = bias
        self.weights = numpy.array([random.random() for i in range(d)])
        self.weights = numpy.array([random.randint(0,1000) for i in range(d)])
        self.weights = self.weights / numpy.sum(self.weights)
        self.error = []

    def output(self,x):
        self.weighted_input = self.bias + numpy.vdot(x,self.weights)
        r = self.activation_function(self.weighted_input)
        #return 1 if r>=0.5 else 0
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
