# Deep Dream GUI
from tkinter import Tk, messagebox, Menu, Label, Entry, Button, StringVar, ttk, Spinbox, filedialog
import PIL.Image

def deep_dream_tk_gui():
    # toolbar functions ********************
    # column one, 'File' ******************
    def load_image():
        file_to_open = filedialog.askopenfile()
        image = PIL.Image.open(file_to_open.name)
        print(file_to_open.name)

    def tb_new():
        messagebox.showinfo(title="Deep-Dream GUI", message='Still working on it.')

    def save_image():

        print(messagebox.askyesno(title='Hungry?', message='Do you want SPAM?'))
        messagebox.showinfo(title="Deep-Dream GUI", message='Still working on it.')


    def tb_open():
        file_to_open = filedialog.askopenfile()
        print(file_to_open.name)

    def close_application():
        exit()

    # column two, 'Edit' ******************
    def tb_undo():
        messagebox.showinfo(title="Deep-Dream GUI", message='Still working on it.')

    def tb_redo():
        messagebox.showinfo(title="Deep-Dream GUI", message='Still working on it.')



    root = Tk()
    root.geometry('450x450+100+200')
    root.title('Deep-Dream GUI')
    toolbar = Menu(root)
    root.config(menu=toolbar)
    # Toolbar Column One ******************************************************
    subMenu1 = Menu(toolbar)
    toolbar.add_cascade(label='File', menu=subMenu1)
    subMenu1.configure(background='whitesmoke')
    # Column One Options
    subMenu1.add_command(label='Load Image', command=load_image)
    subMenu1.add_command(label='Save Image', command=save_image)
    subMenu1.add_separator()
    subMenu1.add_command(label='Exit', command=close_application)


    # Deep Dream Options Interface ****************************************************
    load_img_button = Button(root, text='Load Image', fg='black', bg='silver', command=load_image).pack()
    dd_label_1 = Label(root, text='Choose a layer').pack()
    layer = StringVar()
    combobox = ttk.Combobox(root, textvariable=layer)
    combobox.pack()
    combobox.config(values=('conv2d1', 'conv2d2', 'mixed3a', 'mixed3b', 'mixed4a', 'mixed4b',
                            'mixed4c', 'mixed4d', 'mixed4e', 'mixed5a', 'mixed5b'))

    dd_label_2 = Label(root, text='Choose Iteration Level').pack()
    iterations = StringVar()
    Spinbox(root, from_=1, to=33, textvariable=iterations).pack()

    dd_label_2 = Label(root, text='Choose Repeat Level').pack()
    repeats = StringVar()
    Spinbox(root, from_=1, to=12, textvariable=repeats).pack()


    dd_button = Button(root, text='Deep Dream', fg='black', bg='silver').pack()






    root.mainloop()
deep_dream_tk_gui()



