
from mnist import MNIST
import numpy,random, yaml, re, sys, os
from PIL import Image, ImageDraw
import time
from matplotlib import pyplot as plt
import datetime as dt
from utils import log

class Application():
	classifier = {}
	classifier_name = None
	win = None

	def __init__(self):
		self.load_config()
		self.load_data()
		self.init_test()

	def load_config(self):
		with open("config.yaml", 'r') as ymlfile:
			self.cfg = yaml.load(ymlfile)

	def load_data(self):
		self.mndata = MNIST(self.cfg['folder']['data'])
		self.mndata.load_training()
		self.mndata.load_testing()
		log("data loaded")

	def init_test(self):
		self.tested = 0
		self.success = 0
		self.confusion_matrix = numpy.zeros((10,11), dtype=int)

	def set_window(self, win):
		self.win = win

	def test(self):
		self.tested += 1
		n = random.randint(0,len(self.mndata.test_images)-1)
		expected = self.mndata.test_labels[n]
		output = self.predict(self.mndata.test_images[n])
		if self.win != None:
			self.win.print_matrix(self.mndata.test_images[n], expected)
		self.update_info(expected,output)

	def test_multiple(self, test_number):
		self.load_last_classifier(self.mode.get())
		percent = 1
		for t in range(test_number):
			self.test()
			if t % (test_number // 100) == 0:
				print(percent,'%')
				percent += 1
		print("results numpy.saved in ", self.cfg['folder']['output'] + 'results.txt')

	def load_last_classifier(self, classifier_name):
		l = os.listdir(self.cfg['folder']['classifier'])
		l = list(filter(re.compile(classifier_name + '*').match, l))
		l.sort()
		if len(l) == 0:
			print("Aucune sauvegarde de " + classifier_name + " n'a été enregistrer, essayez :")
			print("python3 main.py " + classifier_name + " train")
			sys.exit()
		else:
			self.load_classifier(l[0])

	def update_info(self,expected, output):
		if output == expected:
			self.success += 1
		if output != '_' and expected != '?':
			self.confusion_matrix[int(expected),int(output)] += 1
		elif expected != '?':
			self.confusion_matrix[int(expected),10] += 1
		numpy.savetxt(self.cfg['folder']['output'] + 'result.txt', self.confusion_matrix, fmt="%d")
		result = open('../output/result.txt', 'a')
		result.write("\nattendue: " + str(expected))
		result.write("\nsortie: " + str(output))
		result.write("\ntest: " + str(self.tested))
		result.write("\nexactitude: " + str(round(self.success/self.tested*100,3)))
		result.close()
		result = open(self.cfg['folder']['output'] + 'result.txt', 'r')
		data = result.read()
		result.close()
		if self.win != None:
			self.win.result.config(text = data)

	def load_classifier(self, classifier_file):
		classifier_save = numpy.load(self.cfg['folder']['classifier'] + classifier_file)
		classifier_name = classifier_file.split('_')[0]
		self.classifier[classifier_name].load_save(classifier_save)
		print("classifier '" + classifier_file + "' loaded")

	def add_classifier(self, name, classifier):
		self.classifier[name] = classifier
		classifier.setConfig(self.cfg['classifier'][name])

	def predict(self,image):
		return self.get_classifier().predict(image)

	def get_classifier(self):
		return self.classifier[self.classifier_name]

	def train(self, save_classifier=False):
		start = time.time()
		self.get_classifier().train(self.mndata.train_images, self.mndata.train_labels,start)
		log("trained in " + str(time.time() - start) + "s")
		filepath = self.cfg['folder']['classifier'] + self.classifier_name + '_' + dt.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + '.npy'
		numpy.save(filepath, self.get_classifier().get_save())
