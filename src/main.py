import tkinter as tk
from gui.application import Application

from classifier.perceptron import PerceptronClassifier

root = tk.Tk()

app = Application(root)
app.master.title("Reconnaissance de caractères")

root.protocol("WM_DELETE_WINDOW", app.on_exit)

app.add_classifier("Perceptron", PerceptronClassifier())

app.mainloop()
