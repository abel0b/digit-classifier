import tkinter as tk
from application import Application

from perceptron import PerceptronClassifier
from multilayer import MultilayerPerceptronClassifier

root = tk.Tk()

app = Application(root)
app.master.title("Reconnaissance de caractères")

root.protocol("WM_DELETE_WINDOW", app.on_exit)

app.add_classifier("Perceptron", PerceptronClassifier())
app.add_classifier("Multilayer perceptron", MultilayerPerceptronClassifier())

app.mainloop()
