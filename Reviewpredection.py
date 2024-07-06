import tkinter as tk
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

class FactorAnalysisApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Factor Analysis Tool")
        self.root.geometry("1000x600")

        self.file_path = r"D:\PA project\survey.xlsx"  # Replace with your file path

        self.create_widgets()
        self.analyze_data()

    def create_widgets(self):
       

        self.back_button = tk.Button(self.root, text="Back", command=self.exit_program)
        self.back_button.pack(side=tk.TOP, pady=10)

        self.figure, self.axes = plt.subplots(1, 2, figsize=(12, 6))

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True)

        self.toolbar = NavigationToolbar2Tk(self.canvas, self.root)
        self.toolbar.update()
        self.canvas.draw()

    def analyze_data(self):
        try:
            df = pd.read_excel(self.file_path)
            data = df.drop(columns=['user_id', 'language', 'platform', 'gender', 'course_completion'])
            data = data.fillna(data.mean())
            scaler = StandardScaler()
            data_scaled = scaler.fit_transform(data)
            n_factors = 5
            fa = FactorAnalysis(n_components=n_factors, random_state=0)
            fa.fit(data_scaled)
            factor_loadings = fa.components_
            loadings_df = pd.DataFrame(factor_loadings, columns=data.columns)

            self.display_factor_loadings(loadings_df)
            self.display_heatmap(loadings_df)
            self.display_line_graph(loadings_df)
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")

    def display_factor_loadings(self, loadings_df):
        print("hi")
        #self.output_text.insert(tk.END, "Factor Loadings:\n")
        #self.output_text.insert(tk.END, loadings_df.to_string())

    def display_heatmap(self, loadings_df):
        sns.heatmap(loadings_df, annot=True, cmap='coolwarm', cbar=True, ax=self.axes[0])
        self.axes[0].set_title('Factor Loadings Heatmap')
        self.axes[0].set_xlabel('Variables')
        self.axes[0].set_ylabel('Factors')

    def display_line_graph(self, loadings_df):
        for i in range(loadings_df.shape[0]):
            self.axes[1].plot(loadings_df.columns, loadings_df.iloc[i], marker='o', label=f'Factor {i+1}')
        self.axes[1].set_title('Factor Loadings Line Graph')
        self.axes[1].set_xlabel('Variables')
        self.axes[1].set_ylabel('Loadings')
        self.axes[1].legend()
        self.axes[1].tick_params(axis='x', rotation=90)

    def exit_program(self):
        self.root.destroy()

root = tk.Tk()
app = FactorAnalysisApp(root)
root.mainloop()
