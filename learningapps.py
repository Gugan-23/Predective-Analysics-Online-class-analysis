import pandas as pd
import statsmodels.api as sm
import tkinter as tk
from tkinter import scrolledtext
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

import warnings
def linear_and_poisson_regression():
    # Read data from Excel file
    file_path = 'D:\\PA project\\app_info.xlsx'
    data = pd.read_excel(file_path)

    # Define independent and dependent variables
    X = data['minInstalls']
    y = data['score']

    # Add constant term for linear regression
    X = sm.add_constant(X)

    # Perform linear regression
    linear_model = sm.OLS(y, X).fit()

    # Perform Poisson regression
    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    # Create plots
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(X['minInstalls'], y, label='Data', s=50, alpha=0.7)
    ax.plot(X['minInstalls'], linear_model.predict(X), color='red', label='Linear Regression')
    ax.plot(X['minInstalls'], poisson_model.predict(X), color='blue', label='Poisson Regression')
    ax.set_xlim(X['minInstalls'].min() - 1000, X['minInstalls'].max() + 1000)
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
    ax.set_xlabel('minInstalls')
    ax.set_ylabel('score')
    ax.set_title('Linear and Poisson Regression')
    ax.legend()

    # Display summary
    output_text.insert(tk.END, "Linear Regression Summary:\n")
    output_text.insert(tk.END, linear_model.summary())
    output_text.insert(tk.END, "\n\nPoisson Regression Summary:\n")
    output_text.insert(tk.END, poisson_model.summary())

    # Embed Matplotlib plot into Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

warnings.filterwarnings("ignore", message="kurtosistest only valid for n>=20", category=UserWarning)
# Create GUI window
root = tk.Tk()
root.title("Regression Analysis")

# Create button to trigger regression analysis
analyze_button = tk.Button(root, text="Analyze", command=linear_and_poisson_regression)
analyze_button.pack()

# Create text area to display results
output_text = scrolledtext.ScrolledText(root, width=100, height=20)
output_text.pack()

# Run GUI
root.mainloop()
