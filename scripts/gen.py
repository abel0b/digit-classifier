

act = 'sgn'

perceptron_train = 'python3 ../src/main.py train Perceptron --activation {2} --it {0} --eta {1} --outputs-folder ./out/ --models-folder ./mod/'

perceptron_test = 'python3 ../src/main.py test Perceptron --activation {2} --it {0} --eta {1} --outputs-folder ./out/ --models-folder ./mod/'


it = [k*100 for k in range(1000)]
eta = [10, 1, 0.1, 0.01, 0.001]

l = ['#!/bin/bash']

def gen_script(act):
    for k in range(len(it)):
        for j in range(len(eta)):
            l.append(perceptron_train.format(it[k], eta[j], act))
            l.append(perceptron_test.format(it[k], eta[j], act))

gen_script('sgn')

with open('script.sh', 'w') as filehandler:
    for i in range(len(l)):
        filehandler.write('\n'+l[i])
        if i % len(l) == 0:
            filehandler.write('\necho ' + str(i / len(l) * 100) + '%')
