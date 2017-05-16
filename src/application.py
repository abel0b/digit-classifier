import tkinter as tk
from mnist import MNIST
from numpy import array, zeros, savetxt
from random import randint
from PIL import Image, ImageDraw
import time
from matplotlib import pyplot as plt
import yaml

class Application(tk.Frame):
	CANVAS_WIDTH, CANVAS_HEIGHT = 224, 224
	WIN_WIDTH = 610
	WIN_HEIGHT = 285
	classifier = {}
	lastx, lasty = 0, 0
	b1down = False

	def __init__(self, master):
		tk.Frame.__init__(self, master)
		self.master = master
		self.load_config()
		self.center_window()
		self.create_widgets()
		self.grid()
		self.mndata = MNIST(self.cfg['folder']['data'])
		self.load_data()
		self.after(0,self.update_screen)
		self.init_test()
		self.img = Image.new("L", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), 'white')
		self.imdraw = ImageDraw.Draw(self.img)
		self.exit = False

	def load_config(self):
		with open("config.yaml", 'r') as ymlfile:
			self.cfg = yaml.load(ymlfile)

	def on_exit(self):
		self.after_cancel(self.id)
		for x,y in enumerate(self.classifier):
			self.classifier[y].close()
		self.master.destroy()

	def center_window(self):
	    ws = self.master.winfo_screenwidth()
	    hs = self.master.winfo_screenheight()
	    x = (ws/2) - (self.WIN_WIDTH/2)
	    y = (hs/2) - (self.WIN_HEIGHT/2)
	    self.master.geometry('%dx%d+%d+%d' % (self.WIN_WIDTH, self.WIN_HEIGHT, x, y))

	def load_data(self):
		self.mndata.load_training()
		self.mndata.load_testing()
		self.print_matrix(self.mndata.test_images[0],self.mndata.test_labels[0])
		self.log("data loaded")

	def log(self, message):
		print(message)

	def print_matrix(self,img,label="_"):
		pi = tk.PhotoImage(width=28,height=28)
		for i in range(28):
			for j in range(28):
				if type(img) == list:
					pi.put("#%02x%02x%02x" % (255-img[i*28+j],255-img[i*28+j],255-img[i*28+j]),(j,i))
				else:
					pi.put("#%02x%02x%02x" % (img[i,j],img[i,j],img[i,j]),(i,j))
		pi = pi.zoom(8,8)
		self.pi = pi
		self.center_canvas.create_image(0, 0, image = pi, anchor = tk.NW)


	def init_test(self):
		self.tested = 0
		self.success = 0
		self.confusion_matrix = zeros((10,11), dtype=int)

	def test(self):
		self.tested += 1
		n = randint(0,len(self.mndata.test_images)-1)
		expected = self.mndata.test_labels[n]
		output = self.predict(self.mndata.test_images[n])
		self.print_matrix(self.mndata.test_images[n], expected)
		self.update_info(expected,output)

	def update_info(self,expected, output):
		if output == expected:
			self.success += 1
		if output != '_' and expected != '?':
			self.confusion_matrix[int(expected),int(output)] += 1
		elif expected != '?':
			self.confusion_matrix[int(expected),10] += 1
		savetxt(self.cfg['folder']['output'] + 'result.txt', self.confusion_matrix, fmt="%d")
		result = open('../output/result.txt', 'a')
		result.write("\nattendue: " + str(expected))
		result.write("\nsortie: " + str(output))
		result.write("\ntest: " + str(self.tested))
		result.write("\nexactitude: " + str(round(self.success/self.tested*100,3)))
		result.close()
		result = open(self.cfg['folder']['output'] + 'result.txt', 'r')
		data = result.read()
		result.close()
		self.result.config(text = data)

	def cancel(self):
		pass

	def create_widgets(self):
		tk.Frame.grid(self)
		#left
		self.left = tk.Frame(self, width=self.CANVAS_WIDTH, height=self.WIN_HEIGHT)
		self.canvas = tk.Canvas(self.left, width=self.CANVAS_WIDTH, height=self.CANVAS_HEIGHT, bg="white")
		self.canvas.pack()
		tk.Button(self.left, text='Effacer', command=self.clear_canvas, width=25).pack()
		tk.Button(self.left, text='Annuler', command=self.cancel, width=25).pack()
		self.canvas.bind("<ButtonPress-1>", self.b1down)
		self.canvas.bind("<ButtonRelease-1>", self.b1up)
		self.canvas.bind("<Motion>", self.motion)
		self.b1down = False
		self.canvas.config(cursor="dot")
		self.left.grid(row=0,column=0, sticky=tk.NW)

		#center
		self.center = tk.Frame(self, width=224,height=self.WIN_HEIGHT)
		self.center_canvas = tk.Canvas(self.center, width=224,height=224,bg="white")
		self.center.grid(row=0,column=1, sticky=tk.NW)
		self.center_canvas.pack(anchor=tk.NW)
		tk.Button(self.center, text='Chiffre aléatoire', command=self.test, width=25).pack()
		tk.Button(self.center, text='Entrainer', command=self.train, width=25).pack()

		#sidebar
		self.sidebar = tk.Frame(self, width=220, height=self.WIN_HEIGHT, padx=5, pady=5)
		self.sidebar.grid(row=0,column=2,sticky=tk.N)
		self.result = tk.Label(self.sidebar, text='', justify=tk.LEFT)
		self.modes = tk.LabelFrame(self.sidebar, text='Classifieur', width=224)
		self.modes.pack(anchor=tk.W)
		self.mode = tk.StringVar()
		self.testing = tk.IntVar()
		tk.Checkbutton(self.sidebar, text='Tester', var=self.testing, command=self.init_test, width=14).pack()
		self.result.pack(anchor=tk.W)

	def grid(self):
		pass

	def clear_canvas(self):
		self.canvas.delete(tk.ALL)
		self.img = Image.new("L", (self.CANVAS_WIDTH, self.CANVAS_HEIGHT), 'white')
		self.imdraw = ImageDraw.Draw(self.img)

	def add_classifier(self, name, classifier):
		self.mode.set(name)
		tk.Radiobutton(self.modes, text=name,variable=self.mode, value=name,width=14).pack(anchor=tk.N)
		classifier.setConfig(self.cfg['classifier'][name])
		self.classifier[name] = classifier

	def get_classifier(self):
		return self.classifier[self.mode.get()]

	def update_screen(self):
		self.id = self.after(50,self.update_screen)
		if self.testing.get() == 1:
			self.test()

	def predict(self,image):
		return self.get_classifier().predict(image)

	def train(self):
		start = time.time()
		self.get_classifier().train(self.mndata.train_images, self.mndata.train_labels,start)
		self.log("trained in " + str(time.time() - start) + "s")


	def b1down(self, event):
		self.b1down = True

	def b1up(self, event):
		self.tested += 1
		self.b1down = False
		self.img = self.img.resize((28,28))
		self.print_matrix(self.img.load())
		image = array([0 for i in range(28*28)])
		for i in range(28):
			for j in range(28):
				image[i*28+j] = 255 - self.img.load()[j,i]
		self.update_info("?",self.predict(image))

	def motion(self, event):
		self.lastx, self.lasty = event.x, event.y
		if self.b1down:
			size = 6
			self.canvas.create_oval(self.lastx-size, self.lasty-size, self.lastx+size, self.lasty+size, fill='black', width=0)
			self.imdraw.ellipse([self.lastx-8, self.lasty-size, self.lastx+size, self.lasty+size],fill='black',outline=None)
