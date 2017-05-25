import numpy
import matplotlib.pyplot as plt

from multilayer import Network

entrees = numpy.array([[0,0],[0,1],[1,0],[1,1]])
valeurs_cibles = numpy.array([[0],[1],[1],[0]])

reseau = Network(2,2,1)
iteration = 100000

reseau.train(entrees, valeurs_cibles, it=iteration, eta=0.1)

for e in entrees:
    print(e[0], " oux ", e[1], "=", reseau.feed_forward(e)[0])

plt.plot(numpy.linspace(0, iteration, num=iteration),reseau.cost)
plt.show()
