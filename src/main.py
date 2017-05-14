import tkinter as tk
from gui.application import Application

from classifier.perceptron import PerceptronClassifier

root = tk.Tk()

app = Application(root)
app.master.title("Reconnaissance de caractères")

app.add_classifier("Perceptron", PerceptronClassifier())

app.mainloop()
