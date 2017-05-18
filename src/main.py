import argparse

from application import Application

from perceptron import PerceptronClassifier
from multilayer import MultilayerPerceptronClassifier

parser = argparse.ArgumentParser()
app = Application()

app.add_classifier("Perceptron", PerceptronClassifier())
app.add_classifier("MultilayerPerceptron", MultilayerPerceptronClassifier())

parser.add_argument('action', choices=['train', 'test'])
parser.add_argument('classifier', choices=app.get_classifier_list())

parser.add_argument('--data-folder', default='./data/')
parser.add_argument('--results-file', default='../output/results.txt')
parser.add_argument('--classifier-folder', default='../output/classifier/')

parser.add_argument('--eta', default=0.1)
parser.add_argument('--iteration', default=100000)

args = parser.parse_args()

app.load_data(args.data_folder)
app.set_classifier(args.classifier)

app.set_options(args.eta, args.iterations)

print(args)

if args.action == 'train':
    app.train(args.classifier_folder)
elif args.action =='test':
    app.test(args.results_file, args.classifier_folder)
