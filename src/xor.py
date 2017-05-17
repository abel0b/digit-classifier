

from multilayer import Network
import matplotlib.pyplot as plt

I = [[0,0],[0,1],[1,0],[1,1]]
O = [[0],[1],[1],[0]]

net = Network(2,2,1)
net.backpropagate(I, O)

for i in I:
    print(i)
    print(net.output(i))

plt.plot(range(100000), net.cost)
plt.show()
