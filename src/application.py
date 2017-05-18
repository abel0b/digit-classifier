from mnist import MNIST
import numpy,random, yaml, re, sys, os
from PIL import Image, ImageDraw
import time
from matplotlib import pyplot as plt
import datetime as dt
from utils import log, print_remaining_time, timer_start

class Application():
	classifiers = {}
	classifier = None

	def load_data(self, data_folder):
		self.mndata = MNIST(data_folder)
		self.mndata.load_training()
		self.mndata.load_testing()
		self.normalize_data()
		log("data loaded")

	def normalize_data(self):
		self.mndata.train_images = numpy.array(self.mndata.train_images, dtype="float64") / 255
		self.mndata.test_images = numpy.array(self.mndata.test_images, dtype="float64") / 255

	def init_test(self):
		self.tested = 0
		self.success = 0
		self.confusion_matrix = numpy.zeros((10,11), dtype=int)

	def test(self, results_file, classifier_folder):
		self.init_test()
		self.load_last_classifier(self.classifier, classifier_folder)
		test_number = len(self.mndata.test_images)
		timer_start()
		for t in range(test_number):
			self.update_statistics(self.mndata.test_labels[t], self.classifier.predict(self.mndata.test_images[t]), results_file)
			print_remaining_time(t,test_number)
		print('les résultats ont été sauvegarder dans ' + results_file)

	def update_statistics(self, expected, output, results_file):
		self.tested += 1
		if output == expected:
			self.success += 1
		if output != '_':
			self.confusion_matrix[int(expected),int(output)] += 1
		numpy.savetxt(results_file, self.confusion_matrix, fmt="%d")
		result = open(results_file, 'a')
		result.write("\nattendue: " + str(expected))
		result.write("\nsortie: " + str(output))
		result.write("\ntest: " + str(self.tested))
		result.write("\nsucces: " + str(self.success))
		result.write("\ntaux de succes: " + str(round(self.success/self.tested*100,3)))
		result.close()
		result = open(results_file, 'r')
		data = result.read()
		print(data)
		result.close()

	def load_last_classifier(self, classifier_name, classifier_folder):
		l = os.listdir('../output/classifier')
		l = list(filter(re.compile(classifier_name + '*').match, l))
		l.sort()
		if len(l) == 0:
			print("Aucune sauvegarde de " + classifier_name + " n'a été enregistrer, essayez :")
			print("python3 main.py train " + classifier_name)
		else:
			self.load_classifier(l[-1], classifier_folder)

	def load_classifier(self, classifier_file, classifier_folder):
		classifier_save = numpy.load(classifier_folder + classifier_file)
		classifier_name = classifier_file.split('_')[0]
		self.classifiers[classifier_name].load_save(classifier_save)
		self.classifier = self.classifiers[classifier_name]
		print("classifier '" + classifier_file + "' loaded")

	def add_classifier(self, name, classifier):
		self.classifiers[name] = classifier

	def set_classifier(self, name):
		self.classifier = name

	def get_classifier_list(self):
		return list(self.classifiers.keys())

	def train(self, classifier_folder):
		start = time.time()
		self.classifiers[self.classifier].train(self.mndata.train_images, self.mndata.train_labels)
		log("trained in " + str(time.time() - start) + "s")
		filepath = classifier_folder + self.classifier + '_' + dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.npy'
		numpy.save(filepath, self.classifiers[self.classifier].get_save())

	def set_options(self, eta, it):
		self.classifiers[self.classifier].eta = eta
		self.classifiers[self.classifier].it = it
