from argparse import Namespace
from abc import abstractmethod

from pickle import dump, load
from utils import log

class DigitClassifier:
    args = Namespace()

    def __init__(self, options):
        self.add_arguments(options)

    def add_arguments(self, options):
        pass

    def save(self, filename):
        with open(filename, 'wb') as filehandler:
            dump(self, filehandler)

    @staticmethod
    def load(filename):
        with open(filename, 'rb') as filehandler:
            log('classifieur', filename, 'chargé')
            return load(filehandler)

    @abstractmethod
    def train(self, images, labels):
        pass

    @abstractmethod
    def predict(self, image):
        pass
