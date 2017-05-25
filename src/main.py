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

# dossiers utiles
parser.add_argument('--data-folder', default='./data/')
parser.add_argument('--results-file', default='../output/results.txt')
parser.add_argument('--classifier-folder', default='../output/classifier/')

# options de configuration
parser.add_argument('--eta', default=0.01)
parser.add_argument('--it', default=1000, type=int)
parser.add_argument('--activation', default='sgn', choices=['sgn', 'sigmoid'])

args = parser.parse_args()
app.init(args)

if args.action == 'train':
    app.train(args.classifier, args.classifier_folder)
elif args.action == 'test':
    app.test(args.classifier, args.classifier_folder, args.results_file)
