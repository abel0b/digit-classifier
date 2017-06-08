import numpy
from struct import unpack
from os.path import isfile

class Mnist:
    def __init__(self, datafolder):
        self.datafolder = datafolder
        self.train_images = datafolder + 'processed/train_images.npy'
        self.train_labels = datafolder + 'processed/train_labels.npy'
        self.test_images = datafolder + 'processed/test_images.npy'
        self.test_labels = datafolder + 'processed/test_labels.npy'

    def preprocess(images, negative = False, bounding_box = False):
        return images / 255

    def load(self, mode):
        if mode == 'train':
            return self.load_training(), (numpy.empty(0), numpy.empty(0))
        elif mode == 'test':
            return (numpy.empty(0), numpy.empty(0)), self.load_testing()
        elif mode =='traintest':
            return self.load_training(), self.load_testing()

    def load_training(self):
        if isfile(self.train_labels) and isfile(self.train_images):
            return numpy.load(self.train_labels), numpy.load(self.train_images)
        else:
            return self.load_labels(self.datafolder + 'raw/train-labels-idx1-ubyte', 'train'), self.load_images(self.datafolder + 'raw/train-images-idx3-ubyte', 'train')

    def load_testing(self):
        if isfile(self.test_labels) and isfile(self.test_images):
            return numpy.load(self.test_labels), numpy.load(self.test_images)
        else:
            return self.load_labels(self.datafolder + 'raw/t10k-labels-idx1-ubyte', 'test'), self.load_images(self.datafolder + 'raw/t10k-images-idx3-ubyte', 'test')

    def load_labels(self, filename, mode):
        with open(filename, 'rb') as filehandler:
            magic, size = unpack('>II', filehandler.read(8))
            labels = numpy.fromfile(filehandler, dtype=numpy.uint8)
            if mode == 'train':
                with open(self.train_labels, 'wb') as filehandler:
                    numpy.save(filehandler, labels)
            elif mode == 'test':
                with open(self.test_labels, 'wb') as filehandler:
                    numpy.save(filehandler, labels)
            return labels

    def load_images(self, filename, mode):
        with open(filename, 'rb') as filehandler:
            magic, size, rows, cols = unpack(">IIII", filehandler.read(16))
            images = numpy.fromfile(filehandler, dtype=numpy.uint8).astype(numpy.float64).reshape(size, 784)
            if mode == 'train':
                with open(self.train_images, 'wb') as filehandler:
                    numpy.save(filehandler, images)
            elif mode == 'test':
                with open(self.test_images, 'wb') as filehandler:
                    numpy.save(filehandler, images)
            return self.preprocess(images)

    @staticmethod
    def print_image(image):
        txt = ''
        for i in range(28):
            txt += '\n'
            for j in range(28):
                if image[i*28+j] >= 0.5:
                    txt += '█'
                else:
                    txt += ' '
        print(txt)
