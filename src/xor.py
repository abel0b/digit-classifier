from multilayer import Network

entrees = [[0,0],[0,1],[1,0],[1,1]]
valeurs_cibles = [[0],[1],[1],[0]]

reseau = Network(2,2,1)

reseau.backpropagate(entrees, valeurs_cibles, it=1000000)

for e in entrees:
    print(reseau.output(e))
