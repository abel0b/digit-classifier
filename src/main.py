from application import Application

from perceptron import MulticlassPerceptronClassifier
from multilayer import MultilayerPerceptronClassifier

classifiers = {
    "MulticlassPerceptron": MulticlassPerceptronClassifier,
    "MultilayerPerceptron": MultilayerPerceptronClassifier
}

app = Application(classifiers)
app.run()
