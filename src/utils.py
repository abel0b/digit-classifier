from math import exp
from time import time
import numpy

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

def time_remaining(start, i, imax):
    if i == 0:
        return '_'
    else:
        return sectostr((imax-i)*(time()-start)/i)

def sigmoid(x):
    return 1/(1+exp(-x))


def sigmoid_prime(x):
    return exp(-x)/(1+exp(-x))**2
