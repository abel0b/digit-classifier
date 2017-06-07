import numpy
import struct
from os.path import isfile

class Mnist:
    def __init__(self, datafolder):
        self.datafolder = datafolder
        self.train_images = datafolder + 'processed/train_images.npy'
        self.train_labels = datafolder + 'processed/train_labels.npy'
        self.test_images = datafolder + 'processed/test_images.npy'
        self.test_labels = datafolder + 'processed/test_labels.npy'

    def preprocess(dataloader, normalize = True, bounding_box = False):
        def get_processed_data(self):
            (labels_train, images_train), (labels_test, images_test) = dataloader(self)
            print(images_train.shape)
            return (labels_train, images_train / 255), (labels_test, images_test / 255)
        return get_processed_data

    @preprocess
    def load(self):
        train, test = self.load_training(), self.load_testing()
        print(train[1].shape, test[1].shape)

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
            magic, n = struct.unpack('>II', filehandler.read(8))
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
            magic, num, rows, cols = struct.unpack(">IIII", filehandler.read(16))
            images = numpy.fromfile(filehandler, dtype=numpy.float64)
            images = images.reshape(images.shape[0] // 784,784)
            if mode == 'train':
                with open(self.train_images, 'wb') as filehandler:
                    numpy.save(filehandler, images)
            elif mode == 'test':
                with open(self.test_images, 'wb') as filehandler:
                    numpy.save(filehandler, images)
            return images

    @staticmethod
    def print_image(image):
        for i in range(28):
            for j in range(28):
                line = ''
                if image[i*28+j] >= 0.5:
                    line += '#'
                else:
                    line += ' '
            print(line)


loader = Mnist('../data/')
(train_label, train_image), (test_label, test_images) = loader.load()
loader.print_image(train_image[0])
