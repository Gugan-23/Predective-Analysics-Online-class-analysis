import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy import stats
import tkinter as tk
from tkinter import scrolledtext
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def analyze_regression():
    
    df = pd.read_excel('D:\\PA project\\survey.xlsx', sheet_name='Sheet1')

    
    X = df[['q1s', 'q2s', 'q3s', 'q4s', 'q5s', 'q6s', 'q7s', 'q8s', 'q9s', 'q10s']]
    y = df['Pass/Fail']

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    y_pred_logistic = logistic_model.predict(X_test)
    accuracy_logistic = accuracy_score(y_test, y_pred_logistic)

    
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)
    y_pred_linear = linear_model.predict(X_test)
    mse_linear = mean_squared_error(y_test, y_pred_linear)


    y_pred_linear = y_pred_linear.reshape(-1, 1)

    # Calculate T-test for coefficients
    t_test_results = stats.ttest_ind(X_test, y_pred_linear)

    # Calculate F-test for overall model significance
    f_statistic, p_value = stats.f_oneway(X_test.values.flatten(), y_pred_linear.flatten())

    # Display results in the GUI
    output_text.insert(tk.END, f"Logistic Regression Accuracy: {accuracy_logistic:.2f}\n")
    output_text.insert(tk.END, f"Linear Regression Mean Squared Error: {mse_linear:.2f}\n")
    output_text.insert(tk.END, "T-test Results:\n")
    output_text.insert(tk.END, "\n".join([f"{col}: {res}" for col, res in zip(X.columns, t_test_results)]))
    output_text.insert(tk.END, "\n")
    output_text.insert(tk.END, f"F-test Statistic: {f_statistic:.2f}, p-value: {p_value:.4f}\n")

    # Plot the results
    plt.figure(figsize=(15, 6))

    # Plot Logistic Regression Predictions
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    plt.plot(range(len(y_test)), logistic_model.predict_proba(X_test)[:, 1], color='red', label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Pass/Fail')
    plt.title('Actual vs. Predicted Values - Logistic Regression')
    plt.legend()

    # Plot Linear Regression Predictions
    plt.subplot(1, 2, 2)
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    plt.plot(range(len(y_test)), y_pred_linear, color='green', label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Pass/Fail')
    plt.title('Actual vs. Predicted Values - Linear Regression')
    plt.legend()

    # Embed the plot into Tkinter window
    canvas = FigureCanvasTkAgg(plt.gcf(), master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# Run GUI

    # Plot the results (omitting for brevity)

# Create GUI window
root = tk.Tk()
root.title("Regression Analysis")

# Create button to trigger analysis
analyze_button = tk.Button(root, text="Analyze", command=analyze_regression)
analyze_button.pack()

# Create text area to display results
output_text = scrolledtext.ScrolledText(root, width=100, height=10)
output_text.pack()
plt.figure(figsize=(10, 6))

    # Plot actual vs. predicted values
    
root.mainloop()
