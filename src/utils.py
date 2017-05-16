from math import tanh

def sectostr(sec):
    sec = int(sec)
    minuts = sec // 60
    seconds = sec % 60
    time = ''
    if minuts > 0:
        time += str(minuts) + 'min'
    time += str(seconds) + 's'
    return time

def activation(x):
    return tanh(x)

def activation_prime(x):
    return 1. + tanh(x)**2

def log(*argv):
    print(*argv)
