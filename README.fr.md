# Reconnaissance des chiffres manuscrits

Implémentation de différents algorithmes pour reconnaître les chiffres de la base de donnée MNIST.

## Classifieurs implémentées

|Nom du classifieur  |Méthode utilisée                          |
|--------------------|------------------------------------------|
|Perceptron          |1 couche de perceptrons                   |
|MultilayerPerceptron|1 couche cachée et 1 couche de perceptrons|


## Utilisation
Il faut choisir le classifieur [nom] qu'on souhaite utiliser parmi les choix ci-dessus.
### entrainement
```python3 main.py train [nom]```
### test
```python3 main.py test [nom]```
