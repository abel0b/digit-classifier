'''
Il s'agit du point d'entrée du programme.

Il s'utilise en ligne de commande de la manière suivante :
    python3 main.py [action] [classifieur] [options]
avec [action] l'action souhaitée :
        - train: pour entrainer le classifieur.
        - test: pour le tester
     [classifieur] désigne l'un des deux classifieur implémentées
        - MulticlassPerceptron
        - MultilayerPerceptron
    [options] permet de configuer certain paramètres


'''


from application import Application

from perceptron import MulticlassPerceptronClassifier
from multilayer import MultilayerPerceptronClassifier

classifiers = {
    "MulticlassPerceptron": MulticlassPerceptronClassifier,
    "MultilayerPerceptron": MultilayerPerceptronClassifier
}

app = Application(classifiers)
app.run()
