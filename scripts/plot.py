
import gen

import matplotlib.pyplot as plt

class Outputs:
    OUTFOLDER = './out/'
    
    @staticmethod
    def read_accuracy(config):
        filename = self.OUTFOLDER + '_'.join([str(t) for t in config])
        with open(filename) as filehandler:
            return float(filehandler.readlines()[1][8:-2])

act = 'sgn'

XE = {}
YE = {}

for e in eta:
    XE[e] = []
    YE[e] = []
    for k in it:
        XE[e].append(k)
        YE[e].append(Outputs.read_accuracy(['Perceptron', act, e, k]))

for e in eta:
    plt.plo(XE[e], YE[e])

plt.show()
