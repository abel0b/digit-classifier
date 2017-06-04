from argparse import Namespace
from abc import abstractmethod

class Classifier:
    args = Namespace()

    def __init__(self, options):
        self.add_arguments(options)

    def add_arguments(self, options):
        pass

    @abstractmethod
    def train(self, images, labels):
        pass

    @abstractmethod
    def predict(self, image):
        pass
