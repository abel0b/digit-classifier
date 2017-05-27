

perceptron = 'python3 main.py train Perceptron --activation {2} --it {0} --eta {1} --confusion-matrix-file ./out/perceptron_{0}_{1}_{2}_confusion.txt --error-file ./out/perceptron_{0}_{1}_{2}_error.npy --error-img ./out/perceptron_{0}_{1}_{2}_error.png --results-file ./out/perceptron_{0}_{1}_{2}_results.txt'

perceptron_test = 'python3 main.py test Perceptron --activation {2} --it {0} --eta {1} --confusion-matrix-file ./out/perceptron_{0}_{1}_{2}_confusion.txt --error-file ./out/perceptron_{0}_{1}_{2}_error.npy --error-img ./out/perceptron_{0}_{1}_{2}_error.png --results-file ./out/perceptron_{0}_{1}_{2}_results.txt'

it = [10, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000]
eta = [5, 1, 0.5, 0.4, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.001, 0.0005]

l = ['#!/bin/bash']

def pgen(act):
    for k in range(len(it)):
        for j in range(len(eta)):
            l.append(perceptron.format(it[k], eta[j], act))
            l.append(perceptron_test.format(it[k], eta[j], act))

pgen('sigmoid')
pgen('sgn')

with open('script.sh', 'w') as fi:
    for i in range(len(l)):
        fi.write('\n'+l[i])
