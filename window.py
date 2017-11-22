from Tkinter import *


class App:

    def __init__(self, master):

        frame = Frame(master)
        frame.pack()

        self.button = Button(frame, text = "DONE" , fg = "red", command = frame.quit)
        self.button.pack(side = BOTTOM)
        self.create_ps = Button(frame, text = "Click to create Phase Space", command=self.welcome)
        self.create_ps.pack(side = LEFT)

    def welcome(self):
        print("and we begin!")
root = Tk()

app = App(root)

root.mainloop()
root.destroy()