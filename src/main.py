from sys import argv
import argparse

from application import Application

from perceptron import PerceptronClassifier
from multilayer import MultilayerPerceptronClassifier

GRAPHICAL = (len(argv) == 1)

app = Application()

app.add_classifier("Perceptron", PerceptronClassifier())
app.add_classifier("MultilayerPerceptron", MultilayerPerceptronClassifier())


if GRAPHICAL:
    import tkinter, gui
    root = tkinter.Tk()
    win = gui.Window(root, app)
    app.set_window(win)
    win.master.title("Reconnaissance de caractères")

    root.protocol("WM_DELETE_WINDOW", win.on_exit)

    win.mainloop()
else:
    parser = argparse.ArgumentParser()
    parser.add_argument('classifier', help="Perceptron or MultilayerPerceptron")
    parser.add_argument('action', help="train or test")
    args = parser.parse_args()
    app.classifier_name = args.classifier
    if args.action == 'train':
        app.train(save_classifier=True)
    elif args.action == 'test':
        app.test_multiple(1000)
