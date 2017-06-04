from application import Application

from perceptron import PerceptronClassifier
from multilayer import MultilayerPerceptronClassifier

classifiers = {
    "Perceptron": PerceptronClassifier,
    "MultilayerPerceptron": MultilayerPerceptronClassifier
}

app = Application(classifiers)
app.run()
