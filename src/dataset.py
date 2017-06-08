import numpy
from struct import unpack
from os.path import isfile

class Mnist:
    def __init__(self, datafolder):
        self.datafolder = datafolder
        self.trainfile = datafolder + 'processed/train.npz'
        self.testfile = datafolder + 'processed/test.npz'

    def preprocess(self, images, negative = False, bounding_box = False):
        return images / 255

    def load(self, train = False, test = False):
        if train:
            return self.load_training(), (numpy.empty(0), numpy.empty(0))
        elif test:
            return (numpy.empty(0), numpy.empty(0)), self.load_testing()
        else:
            return self.load_training(), self.load_testing()

    def load_training(self):
        if isfile(self.trainfile):
            arr = numpy.load(self.trainfile)
            return arr['labels'], arr['images']
        else:
            labels, images = self.load_labels(self.datafolder + 'raw/train-labels-idx1-ubyte', 'train'), self.load_images(self.datafolder + 'raw/train-images-idx3-ubyte', 'train')
            self.save(labels, images, self.trainfile)
            return labels, images

    def load_testing(self):
        if isfile(self.testfile):
            arr = numpy.load(self.testfile)
            return arr['labels'], arr['images']
        else:
            labels, images = self.load_labels(self.datafolder + 'raw/t10k-labels-idx1-ubyte', 'test'), self.load_images(self.datafolder + 'raw/t10k-images-idx3-ubyte', 'test')
            self.save(labels, images, self.testfile)
            return labels, images

    def save(self, labels, images, filename):
        numpy.savez_compressed(filename, labels, images, 'labels', 'images')

    def load_labels(self, filename, mode):
        with open(filename, 'rb') as filehandler:
            magic, size = unpack('>II', filehandler.read(8))
            labels = numpy.fromfile(filehandler, dtype=numpy.uint8)
            return labels

    def load_images(self, filename, mode):
        with open(filename, 'rb') as filehandler:
            magic, size, rows, cols = unpack(">IIII", filehandler.read(16))
            images = numpy.fromfile(filehandler, dtype=numpy.uint8).astype(numpy.float64).reshape(size, 784)
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
