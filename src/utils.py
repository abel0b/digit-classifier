from math import exp
from time import time, sleep
import numpy
from sys import stdout
from scipy.special import expit

start = time()


def sectostr(sec):
    sec = int(sec)
    minuts = sec // 60
    seconds = sec % 60
    time = ''
    if minuts > 0:
        time += str(minuts) + 'min'
    time += str(seconds) + 's'
    return time


def log(*argv):
    print(*argv)


def timer_start():
    start = time()


def print_remaining_time(i, imax):
    if imax > 100:
        n = 100
    else:
        n = 1
    ti = i // (imax // n)
    if i > 0 and i % (imax // n) == 0:
        msg = str(ti) + '% : ' + sectostr((imax - i) * (time() - start) / i) + ' restantes' + ' ' * 3
        stdout.write("\r" + msg)
        stdout.flush()
        # print(msg)
    if i == imax - 1:
        stdout.write("\r" + " " * 20)
        print('')


def sgn(x):
    return 1 if x >= 0 else 0

def tanh(x):
    return numpy.tanh(x)

def tanh_prime(x):
    return 1. + numpy.tanh(x)**2

def ntanh(x):
    return 0.5*(numpy.tanh(x)+1)

def ntanh_prime(x):
    return 0.5*(1.+numpy.tanh(x)**2)

def sigmoid(x):
    return expit(x)

def sigmoid_prime(x):
    return sigmoid(x) * (1 - sigmoid(x))
