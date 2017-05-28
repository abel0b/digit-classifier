from mnist import MNIST
import numpy
import random
import yaml
import re
import sys
import os
from PIL import Image, ImageDraw
import time
from matplotlib import pyplot as plt
import datetime as dt
from utils import log, print_remaining_time, timer_start


class Application:
    classifiers = {}
    classifier = None

    def run(self, args):
        self.classifier = self.classifiers[args.classifier](args)
        self.load_data(args.data_folder, args.action)
        if args.action == 'train':
            self.train(args.classifier, args.models_folder)
        elif args.action == 'test':
            self.test(args.classifier, args.outputs_folder, args.models_folder)

    def load_data(self, data_folder, action):
        self.mndata = MNIST(data_folder)
        if action == 'train':
            self.mndata.load_training()
        elif action == 'test':
            self.mndata.load_testing()
        self.normalize_data()
        log("données chargées")

    def normalize_data(self):
        self.mndata.train_images = numpy.array(self.mndata.train_images, dtype="float64") / 255
        self.mndata.test_images = numpy.array(self.mndata.test_images, dtype="float64") / 255

    def load_last_classifier(self, classifier_name, classifier_folder):
        l = os.listdir(classifier_folder)
        l = list(filter(re.compile(classifier_name + '*').match, l))
        l.sort()
        if len(l) == 0:
            log("Aucune sauvegarde de " + classifier_name + " n'a été enregistrer, essayez :")
            log("python3 main.py train " + classifier_name)
        else:
            self.load_classifier(l[-1], classifier_folder)

    def load_classifier(self, classifier_file, classifier_folder):
        classifier_save = numpy.load(classifier_folder + classifier_file)
        classifier_name = classifier_file.split('_')[0]
        self.classifier.load_save(classifier_save)
        log("classifieur '" + classifier_file + "' chargé")

    def init_test(self):
        self.tested = 0
        self.success = 0
        self.wrong_class = 0
        self.ambigu = 0
        self.digit_success = numpy.array([0 for i in range(10)])
        self.digits = numpy.array([0 for i in range(10)])
        self.confusion_matrix = numpy.zeros((10, 11), dtype=int)

    def test(self, classifier_name, outputs_folder, models_folder):
        self.init_test()
        self.load_last_classifier(classifier_name, models_folder)
        test_number = len(self.mndata.test_images)
        timer_start()
        print(outputs_folder)
        for t in range(test_number):
            self.update_statistics(self.mndata.test_labels[t], self.classifier.predict(
                self.mndata.test_images[t]), outputs_folder + 'results.txt', outputs_folder + 'confusion.txt')
            print_remaining_time(t, test_number)
        print(numpy.loadtxt(outputs_folder + 'confusion.txt').astype(int))
        with open(outputs_folder + 'results.txt', 'r') as results_file:
            data = results_file.read()
        print(data)
        plt.bar(range(10), 100*numpy.divide(self.digit_success,self.digits))
        plt.xlabel("chiffre")
        plt.ylabel("taux de succes")
        plt.xticks(range(10))
        plt.yticks(range(0,101,10))
        plt.savefig(outputs_folder + 'results.png')

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
            for i in range(10):
                result.write('\n'+str(self.digit_success[i]))
                result.write('\n'+str(self.digits[i]))

    def add_classifier(self, name, classifier):
        self.classifiers[name] = classifier

    def get_classifier_list(self):
        return list(self.classifiers.keys())

    def train(self, classifier_name, models_folder):
        start = time.time()
        self.classifier.train(self.mndata.train_images, self.mndata.train_labels)
        log("entrainé en " + str(time.time() - start) + "s")
        filepath = models_folder + classifier_name + '_' + \
            dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.npy'
        numpy.save(filepath, self.classifier.get_save())
