from math import exp
from time import time, sleep
import numpy
import sys

global start

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
    n = 100
    ti = i // (imax // n)
    if i > 0 and i % (imax//n) == 0:
        msg = str(ti) + '% : ' + sectostr((imax-i)*(time()-start)/i) + ' restantes'
        print(msg)

def sigmoid(x):
    return 1/(1+exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))
