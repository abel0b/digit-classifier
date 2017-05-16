import tkinter as tk
from sys import argv
import argparse

from application import Application

from perceptron import PerceptronClassifier
from multilayer import MultilayerPerceptronClassifier

root = tk.Tk()

app = Application(root, argv)
app.master.title("Reconnaissance de caractères")

root.protocol("WM_DELETE_WINDOW", app.on_exit)

app.add_classifier("Perceptron", PerceptronClassifier())
app.add_classifier("MultilayerPerceptron", MultilayerPerceptronClassifier())

if len(argv) == 1:
    app.mainloop()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('classifier', help="Perceptron or MultilayerPerceptron")
    parser.add_argument('action', help="train or test")
    args = parser.parse_args()
    app.mode.set(args.classifier)
    if args.action == 'train':
        app.train(save_classifier=True)
    elif args.action == 'test':
        app.test_multiple(1000)
