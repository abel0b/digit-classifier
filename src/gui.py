
import tkinter
from mnist import MNIST
import numpy,random, yaml, re, sys, os
from PIL import Image, ImageDraw
import time
from matplotlib import pyplot as plt
import datetime as dt

class Window(tkinter.Frame):
    CANVAS_WIDTH, CANVAS_HEIGHT = 224, 224
    WIN_HEIGHT = 310
    lastx, lasty = 0, 0
    b1down = False

    def __init__(self, master, app):
        tkinter.Frame.__init__(self, master)
        self.master = master
        master.resizable(width=False, height=False)
        self.app = app
        self.create_widgets()
        self.grid()
        self.after(0,self.update_screen)
        self.app.init_test()
        self.img = Image.new("L", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), 'white')
        self.imdraw = ImageDraw.Draw(self.img)
        self.exit = False
        i = 0
        for name in app.classifier:
            tkinter.Radiobutton(self.modes, text=name,variable=self.mode, value=name).grid(row=i,sticky=tkinter.W)
            i += 1
        self.mode.set(app.classifier_name)

    def on_exit(self):
        for x,y in enumerate(self.app.classifier):
            self.app.classifier[y].close()
        self.master.destroy()

    def log(self, message):
        print(message)

    def print_matrix(self,img,label="_"):
        pi = tkinter.PhotoImage(width=28,height=28)
        for i in range(28):
            for j in range(28):
                if type(img) == list:
                    pi.put("#%02x%02x%02x" % (255-img[i*28+j],255-img[i*28+j],255-img[i*28+j]),(j,i))
                else:
                    pi.put("#%02x%02x%02x" % (img[i,j],img[i,j],img[i,j]),(i,j))
        pi = pi.zoom(8,8)
        self.pi = pi
        self.center_canvas.create_image(0, 0, image = pi, anchor = tkinter.NW)

    def update_screen(self):
        self.id = self.after(50,self.update_screen)
        if self.app.classifier_name != self.mode.get():
            self.app.classifier_name = self.mode.get()
        if self.testing.get() == 1:
            self.app.test_random()

    def menu_load_classifier(self):
        self.top = tkinter.Toplevel(width=300,height=300)
        self.top.grab_set()
        self.top.title("Charger un classifieur")
        classifiers = os.listdir(self.app.cfg['folder']['classifier'])
        self.list_classifiers = tkinter.Listbox(self.top, width=300)
        for classifier in classifiers:
            self.list_classifiers.insert(tkinter.END, classifier)
        self.list_classifiers.pack()
        tkinter.Button(self.top, text="Annuler", command=self.top.destroy).pack()
        tkinter.Button(self.top, text="Charger", command=self.load_classifier).pack()

    def load_classifier(self):
        self.app.load_classifier(self.list_classifiers.get(tkinter.ACTIVE))

    def create_widgets(self):
        tkinter.Frame.grid(self)
        self.grid_columnconfigure(0, weight=1, uniform="fred")
        self.left = tkinter.Frame(self)
        self.canvas = tkinter.Canvas(self.left, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="white")
        self.canvas.pack()
        tkinter.Button(self.left, text='Effacer', command=self.clear_canvas, width=25).pack()
        self.canvas.bind("<ButtonPress-1>", self.b1down)
        self.canvas.bind("<ButtonRelease-1>", self.b1up)
        self.canvas.bind("<Motion>", self.motion)
        self.b1down = False
        self.canvas.config(cursor="dot")
        self.left.grid(row=0,sticky=tkinter.NW)

        #center
        self.center = tkinter.Frame(self)
        self.center.grid(row=0, column=1, sticky=tkinter.NW)
        self.center_canvas = tkinter.Canvas(self.center, height=224, width=224, bg="white")
        self.center_canvas.grid(row=0, sticky=tkinter.EW)
        tkinter.Button(self.center, text='Chiffre aléatoire', command=self.app.test).grid(row=1, sticky=tkinter.EW)
        tkinter.Button(self.center, text='Entrainer', command=self.app.train).grid(row=2, sticky=tkinter.EW)
        tkinter.Button(self.center, text='Charger', command=self.menu_load_classifier).grid(row=3,sticky=tkinter.EW)

        #sidebar
        self.sidebar = tkinter.Frame(self, padx=5, pady=5)
        self.sidebar.grid(row=0, column=2, sticky=tkinter.NW+tkinter.SE)
        self.result = tkinter.Label(self.sidebar, text='')
        self.modes = tkinter.LabelFrame(self.sidebar, text='Classifieur')
        self.modes.grid(row=0, sticky=tkinter.NW+tkinter.E)
        self.mode = tkinter.StringVar()
        self.testing = tkinter.IntVar()
        tkinter.Checkbutton(self.sidebar, text='Tester', var=self.testing, command=self.app.init_test).grid(row=1, sticky=tkinter.W)
        self.result.grid(row=2)

    def grid(self):
        pass

    def clear_canvas(self):
        self.canvas.delete(tkinter.ALL)
        self.img = Image.new("L", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), 'white')
        self.imdraw = ImageDraw.Draw(self.img)

    def get_classifier(self):
        return self.classifier[self.mode.get()]

    def set_classifier(self):
        self.app.classifier_name = self.mode.get()

    def b1down(self, event):
        self.b1down = True

    def b1up(self, event):
        self.app.tested += 1
        self.b1down = False
        self.img = self.img.resize((28,28))
        self.print_matrix(self.img.load())
        image = numpy.array([0 for i in range(28*28)])
        for i in range(28):
            for j in range(28):
                image[i*28+j] = 255 - self.img.load()[j,i]
        self.app.update_info("?",self.app.predict(image))

    def motion(self, event):
        self.lastx, self.lasty = event.x, event.y
        if self.b1down:
            size = 6
            self.canvas.create_oval(self.lastx-size, self.lasty-size, self.lastx+size, self.lasty+size, fill='black', width=0)
            self.imdraw.ellipse([self.lastx-8, self.lasty-size, self.lastx+size, self.lasty+size],fill='black',outline=None)
