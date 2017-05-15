import tkinter as tk
from application import Application

from perceptron import PerceptronClassifier

root = tk.Tk()

app = Application(root)
app.master.title("Reconnaissance de caractères")

root.protocol("WM_DELETE_WINDOW", app.on_exit)

app.add_classifier("Perceptron", PerceptronClassifier())

app.mainloop()
