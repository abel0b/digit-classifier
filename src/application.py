import numpy
import random
import time
from matplotlib import pyplot as plt
import datetime as dt
import argparse
import os
import re
import pickle

from utils import log, print_remaining_time, timer_start

from classifier import DigitClassifier
from dataset import Mnist

import sys

class Application:
    """
        Application
    """
    parser = {}

    '''
    Initialise l'application
    '''
    def __init__(self, classifiers):
        self.classifiers = classifiers

        parser = argparse.ArgumentParser()
        parser.add_argument('action', choices=['train', 'test'])

        subparsers = parser.add_subparsers(dest = 'classifier')

        for name, classifier in self.classifiers.items():
            self.parser[name] = subparsers.add_parser(name)

        name = parser.parse_known_args()[0].classifier

        config_group = self.parser[name].add_argument_group('config')

        self.classifier = self.classifiers[name](config_group) # le classifieur est instancié

        self.config = sorted({ name: value for (name, value) in parser.parse_known_args()[0]._get_kwargs() if name in [action.dest for action in config_group._group_actions]}.items(), key=lambda t: t[0]) # range les options dans l'ordre lexicographique

        folders = self.parser[name].add_argument_group('folders')
        folders.add_argument('--data-folder', default='../data/')
        folders.add_argument('--models-folder', default='../models/')
        folders.add_argument('--outputs-folder', default='../outputs/')

        self.args = parser.parse_args(namespace = self.classifier.args)


        self.classifier.init()
        self.load_data(self.args.data_folder, self.args.action)

    '''
    Lance l'application
    '''
    def run(self):
        if self.args.action == 'train':
            self.train(self.args.classifier, self.args.models_folder)
        elif self.args.action == 'test':
            self.test(self.args.classifier, self.args.outputs_folder, self.args.models_folder)

    '''
    Charge les données d'entrainement ou de test suivant l'actoin souhaitée
    '''
    def load_data(self, data_folder, action):
        mnist = Mnist(data_folder)
        (self.train_labels, self.train_images), (self.test_labels, self.test_images) = mnist.load(action == 'train', action == 'test')
        log("données chargées")

    '''
    Retourne le classifieur entrainée demandé s'il existe, sinon renvoie une Exception.
    '''
    def load_model(self):
        try:
            return DigitClassifier.load(self.model_file + '.pkl')
        except FileNotFoundError:
            log("Le fichier " + self.model_file + ".pkl n'existe pas.")
            sys.exit()

    '''
    Initialise les données de test afin d'évaluer le modèle.
    '''
    def init_test(self):
        self.tested = 0
        self.success = 0
        self.wrong_class = 0
        self.ambigu = 0
        self.digit_success = numpy.array([0 for i in range(10)])
        self.digits = numpy.array([0 for i in range(10)])
        self.confusion_matrix = numpy.zeros((10, 11), dtype=int)

    '''
    Test le modèle demandé, enregistre les résultats dans le dossier spécifié et affiche quelques résultats.
    '''
    def test(self, classifier_name, outputs_folder, models_folder):
        self.init_test()
        self.classifier = self.load_model()
        test_number = len(self.test_images)
        timer_start()
        for t in range(test_number):
            self.update_statistics(self.test_labels[t], self.classifier.predict(
                self.test_images[t]), self.output_file + '.txt', outputs_folder + 'confusion.txt')
            print_remaining_time(t, test_number)
        print(numpy.loadtxt(outputs_folder + 'confusion.txt').astype(int))
        with open(self.output_file + '.txt', 'r') as results_file:
            data = results_file.read()
        print(data)
        plt.bar(range(10), 100*numpy.divide(self.digit_success,self.digits))
        plt.xlabel("chiffre")
        plt.ylabel("taux de succes")
        plt.xticks(range(10))
        plt.yticks(range(0,101,10))
        plt.savefig(outputs_folder + 'results.png')

    '''
    Met à jour les statistiques suivant la réussite ou l'échec du classifieur à classer un chiffre.
        'expected' désigne le chiffre attendu
        'output' désigne le chiffre prédit
    '''
    def update_statistics(self, expected, output, results_file, confusion_matrix_file):
        self.tested += 1
        self.digits[expected] += 1
        if output == expected:
            self.success += 1
            self.digit_success[expected] += 1
        if output != -1:
            self.confusion_matrix[int(expected), int(output)] += 1
            if output != expected:
                self.wrong_class += 1
        else:
            self.confusion_matrix[expected, 10] += 1
            self.ambigu += 1
        numpy.savetxt(confusion_matrix_file, self.confusion_matrix)
        with open(results_file, 'w') as result:
            result.write("  test: " + str(self.tested))
            result.write("\nsucces: " + str(round(self.success / self.tested * 100, 3)) + "%")
            result.write("\nerreur: " + str(round(self.wrong_class / self.tested * 100, 3)) + "%")
            if self.ambigu != 0:
                result.write("\nambigu: " + str(round(self.ambigu / self.tested * 100, 3)) + "%")

    '''
    Effectue l'entrainement du classifieur et son enregistrement dans un fichier.
    '''
    def train(self, classifier_name, models_folder):
        start = time.time()
        self.classifier.train(self.train_images, self.train_labels)
        log("entrainé en " + str(time.time() - start) + "s")

        self.classifier.save(self.model_file + '.pkl')

    '''
    Retourne le nom du fichier servant à enregistrer le classifieur demandé.
    '''
    @property
    def model_file(self):
        return self.args.models_folder + self.args.classifier + '_' + '_'.join(str(value) for (name, value) in self.config)

    '''
    Retourne le nom du fichier servant à enregistrer les résultats du classifieur demandé.
    '''
    @property
    def output_file(self):
        return self.args.outputs_folder + self.args.classifier + '_' + '_'.join(str(value) for (name, value) in self.config)
