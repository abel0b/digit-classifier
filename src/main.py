import argparse

from application import Application

from perceptron import PerceptronClassifier
from multilayer import MultilayerPerceptronClassifier

parser = argparse.ArgumentParser()
app = Application()

app.add_classifier("Perceptron", PerceptronClassifier)
app.add_classifier("MultilayerPerceptron", MultilayerPerceptronClassifier)

parser.add_argument('action', choices=['train', 'test'])
parser.add_argument('classifier', choices=app.get_classifier_list())

# dossiers et fichiers utilisés
parser.add_argument('--data-folder', default='./data/')
parser.add_argument('--classifier-folder', default='../output/classifier/')
parser.add_argument('--results-file', default='../output/results.txt')
parser.add_argument('--confusion-matrix-file', default='../output/confusion.txt')
parser.add_argument('--error-file', default='../output/error.npy')
parser.add_argument('--error-img', default='../output/error.png')

# options de configuration
parser.add_argument('--eta', default=0.01, type=float)
parser.add_argument('--it', default=1000, type=int)
parser.add_argument('--activation', default='sgn', choices=['sgn', 'sigmoid', 'tanh', 'ntanh'])

args = parser.parse_args()
app.run(args)
