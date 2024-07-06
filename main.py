import tkinter as tk
from tkinter import messagebox

def import_numpy():
    import PA

def import_pandas():
    import Reviewpredection

def import_matplotlib():
    import learningapps
def import_matplotlib1():
    import clustering

root = tk.Tk()
root.title("Package Importer")

btn_numpy = tk.Button(root, text="Logistic regression and linear regression", command=import_numpy)
btn_numpy.pack()

btn_pandas = tk.Button(root, text="R^2 and MSE", command=import_pandas)
btn_pandas.pack()

btn_matplotlib = tk.Button(root, text="Poisson regression and linear regression", command=import_matplotlib)
btn_matplotlib.pack()

btn_matplotlib1 = tk.Button(root, text="Clustering for the websites", command=import_matplotlib1)
btn_matplotlib1.pack()


root.mainloop()
