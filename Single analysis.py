import tkinter as tk
from tkinter import messagebox, simpledialog
from datetime import datetime
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.ar_model import AutoReg
from scipy.stats import f
import matplotlib.pyplot as plt

class StudentAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("Student Analysis App")

        self.label_name = tk.Label(master, text="Enter your name:")
        self.label_name.pack()

        self.entry_name = tk.Entry(master)
        self.entry_name.pack()

        self.label_age = tk.Label(master, text="Enter your age:")
        self.label_age.pack()

        self.entry_age = tk.Entry(master)
        self.entry_age.pack()

        self.button_marks = tk.Button(master, text="Enter Marks", command=self.get_marks)
        self.button_marks.pack()

        self.label_forecast = tk.Label(master, text="Forecast:")
        self.label_forecast.pack()

        self.forecast_text = tk.Label(master, text="")
        self.forecast_text.pack()

        self.label_result = tk.Label(master, text="")
        self.label_result.pack()

    def get_marks(self):
        self.name = self.entry_name.get()
        self.age = self.entry_age.get()

        self.marks = []
        for i in range(1, 5):
            mark = simpledialog.askinteger("Marks", f"Enter marks for Subject {i} (1 for correct, 0 for incorrect):")
            if mark is None:
                return
            self.marks.append(mark)

        self.forecast_next()
        self.determine_result()
        self.print_user_details()
        self.perform_f_test()
        self.perform_pca()
        self.plot_graph()

    def forecast_next(self):
        model = AutoReg(self.marks, lags=1)
        model_fit = model.fit()
        self.forecast_value = model_fit.predict(len(self.marks), len(self.marks))[0]
        self.forecast_text.config(text=f"Forecast for the next period based on autoregression: {self.forecast_value}")

    def determine_result(self):
        average_mark = sum(self.marks) / len(self.marks)
        if average_mark >= 0.5:
            self.label_result.config(text="Congratulations! You passed.")
        else:
            self.label_result.config(text="Sorry, you failed.")

    def print_user_details(self):
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        details = f"\nUser's Details:\nName: {self.name}\nAge: {self.age}\nCurrent Time: {current_time}"
        messagebox.showinfo("User Details", details)

    def perform_f_test(self):
        # Compute the proportion of ones in the data
        p = np.mean(self.marks)

        # Compute the observed variance
        var_observed = p * (1 - p)

        # Expected variance under the null hypothesis
        expected_variance = 0.25  # For example, assuming equal variance for 1s and 0s

        # Degrees of freedom for the numerator and denominator
        DF_numerator = 1
        DF_denominator = len(self.marks) - 1

        # Compute the F-statistic
        F_statistic = var_observed / expected_variance

        # Compute the p-value
        p_value = f.cdf(F_statistic, DF_numerator, DF_denominator)
        messagebox.showinfo("F-statistic and p-value", f"F-statistic: {F_statistic}\n p-value: {p_value}")

    def perform_pca(self):
        # Convert binary data into numpy array
        X = np.array([self.marks]).T  # Transpose to have rows as observations

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Perform PCA
        pca = PCA()
        pca.fit(X_scaled)

        # Check if total variance is zero
        total_var = np.sum(pca.explained_variance_)
        if total_var == 0:
            explained_variance_ratio = [0] * len(pca.explained_variance_ratio_)
        else:
            explained_variance_ratio = pca.explained_variance_ratio_

        # Print the explained variance ratio
        pca_result = f"Explained Variance Ratio: {explained_variance_ratio}"
        messagebox.showinfo("PCA Results", pca_result)

    def plot_graph(self):
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        # Plot marks
        axs[0].plot(range(1, len(self.marks) + 1), self.marks, marker='o', linestyle='-')
        axs[0].set_title('Marks')
        axs[0].set_xlabel('Subject Number')
        axs[0].set_ylabel('Mark (1 or 0)')

        # Plot PCA explained variance ratio
        X = np.array([self.marks]).T
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        explained_variance_ratio = pca.explained_variance_ratio_

        axs[1].bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
        axs[1].set_title('PCA Explained Variance Ratio')
        axs[1].set_xlabel('Principal Component')
        axs[1].set_ylabel('Explained Variance Ratio')

        plt.tight_layout()
        plt.show()

root = tk.Tk()
app = StudentAnalysisApp(root)
root.mainloop()
