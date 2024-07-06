from tkinter import ttk, messagebox, Tk, Frame, Button, Label, Entry, scrolledtext
import csv
import tkinter as tk
import webbrowser
import datetime
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import mysql.connector
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from tkinter import PhotoImage
from matplotlib.backends.backend_tkagg import NavigationToolbar2Tk
from sklearn.decomposition import FactorAnalysis
from PIL import Image, ImageTk
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
global output_text5
output_text5 = None

global a1
global output_text
global accuracy_logistic, mse_linear, t_test_results, f_statistic, p_value, X

accuracy_logistic = 0.0
mse_linear = 0.0
t_test_results = []
f_statistic = 0.0
p_value = 0.0
X = pd.DataFrame()

def analyze_regression():
    global accuracy_logistic, mse_linear, t_test_results, f_statistic, p_value, X

    df = pd.read_excel('D:\\PA project\\survey.xlsx', sheet_name='Sheet1')

    # Clean the data
    X = df[['q1s', 'q2s', 'q3s', 'q4s', 'q5s', 'q6s', 'q7s', 'q8s', 'q9s', 'q10s']]
    y = df['Pass/Fail']
    df = df.dropna()

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Split data into train and test sets
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

    # Create a new frame for the plot
    plot_frame = Frame(root)
    plot_frame.place(x=0, y=0, width=1000, height=400)  # Adjust frame size here

    back_button = Button(plot_frame, text='Back', command=lambda: back_to_a1(plot_frame))
    back_button.pack(side=tk.BOTTOM, pady=10)
    
    # Add canvas for the plot
    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # Add a button to show the results
    show_results_button = Button(plot_frame, text='Show Values', command=regal)
    show_results_button.pack(side=tk.BOTTOM, pady=10)
    plt.figure(figsize=(8, 4))

    # Plot Logistic Regression Predictions
    plt.subplot(1, 2, 1)
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    plt.plot(range(len(y_test)), logistic_model.predict_proba(X_test)[:, 1], color='red', label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Pass/Fail')
    plt.title('Actual vs. Predicted Values - Logistic Regression')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual')
    plt.plot(range(len(y_test)), y_pred_linear, color='green', label='Predicted')
    plt.xlabel('Index')
    plt.ylabel('Pass/Fail')
    plt.title('Actual vs. Predicted Values - Linear Regression')
    plt.legend()

    # Create a new frame for the plot
    plot_frame = Frame(root)
    plot_frame.place(x=0, y=0, width=1000, height=500)

    back_button = Button(plot_frame, text='Back', command=lambda: back_to_a1(plot_frame))
    back_button.pack(side=tk.BOTTOM, pady=10)

    # Add canvas for the plot
    canvas = FigureCanvasTkAgg(plt.gcf(), master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
def pro():
    global logged_in_user

    for widget in root.winfo_children():
        widget.destroy()

    p1 = Frame(root, width=2000, height=500, bg='light blue')
    p1.place(x=0, y=0)

    heading = Label(p1, text='Profile', fg='black', bg='light blue', font=('Times New Roman', 23, 'bold'))
    heading.place(x=300, y=20)

    if logged_in_user:
        user_info = f"Username: {logged_in_user[1]}\n\nPassword: {logged_in_user[2]}\n\nGender: {logged_in_user[3]}"
        user_info_label = Label(p1, text=user_info, fg='black', bg='light blue', font=('Times New Roman', 14))
        user_info_label.place(x=200, y=80)
    
    Button(p1, width=20, pady=7, text='Log out', bg='brown', font=('Times New Roman', 10, 'bold'), fg='white', border=3, command=logout).place(x=220, y=250)
    Button(p1, width=20, pady=7, text='Home', bg='brown', font=('Times New Roman', 10, 'bold'), fg='white', border=3, command=search).place(x=220, y=300)
    img = PhotoImage(file='prof.png',width=1000,height=700)
    Label(p1,width=1000,height=700,image=img, bg='light blue').place(x=500,y=150)
    root.mainloop()

def regal():
    w = Frame(root)
    w = Frame(root, width=2000, height=500, bg='white')
    w.place(x=0, y=0)
    frame = Frame(w, width=42000, height=4500, bg='white').place(x=-100, y=-100)
   
    global output_text
    
    output_text = scrolledtext.ScrolledText(w, width=1000, height=10)
    output_text.pack()

    output_text.insert(tk.END, f"Logistic Regression Accuracy: {accuracy_logistic:.2f}\n")
    output_text.insert(tk.END, f"Linear Regression Mean Squared Error: {mse_linear:.2f}\n")
    output_text.insert(tk.END, "T-test Results:\n")
    output_text.insert(tk.END, "\n".join([f"{col}: {res}" for col, res in zip(X.columns, t_test_results)]))
    output_text.insert(tk.END, "\n")
    output_text.insert(tk.END, f"F-test Statistic: {f_statistic:.2f}, p-value: {p_value:.4f}\n")

# Regression and logistic analysis end
#poisson regression------------------------------
def pos():
    global frame  
    if 'frame' in globals():
        frame.destroy()  

    linear_and_poisson_regression()

def linear_and_poisson_regression():

    
    global frame
    
    w1 = Frame(root)
    w1 = Frame(root, width=2000, height=500, bg='white')
    w1.place(x=0, y=0)
    frame = Frame(w1, width=42000, height=4500, bg='white')
    frame.place(x=-100, y=-100)
   
    file_path = 'D:\\PA project\\app_info.xlsx'
    data = pd.read_excel(file_path)
    data= data.drop_duplicates()


    X = data['minInstalls']
    y = data['score']
    if X is not None and y is not None and len(X) == len(y):
        # Check for missing or NaN values in X and y
        if X.isnull().values.any() or np.isnan(X.values).any() or np.isnan(y).any():
            # Handle missing or NaN values
            # Example: Remove rows with missing or NaN values
            X = X.dropna()
            X = X.replace([np.inf, -np.inf], np.nan).dropna()
            y = y.dropna()
        
        # Fit the model
        linear_model = sm.OLS(y, X).fit()
        # Continue with the rest of your analysis...
    else:
        print("Variables X and y are not properly defined or have different lengths.")


    X = sm.add_constant(X)

    linear_model = sm.OLS(y, X).fit()

    poisson_model = sm.GLM(y, X, family=sm.families.Poisson()).fit()

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.scatter(X['minInstalls'], y, label='Data', s=50, alpha=0.7)
    ax.plot(X['minInstalls'], linear_model.predict(X), color='red', label='Linear Regression')
    ax.plot(X['minInstalls'], poisson_model.predict(X), color='blue', label='Poisson Regression')
    ax.set_xlim(X['minInstalls'].min() - 1000, X['minInstalls'].max() + 1000)
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
    ax.set_xlabel('minInstalls')
    ax.set_ylabel('score')
    ax.set_title('Linear and Poisson Regression')
    ax.legend()

    output_text1 = scrolledtext.ScrolledText(frame, width=100, height=15)
    output_text1.place(x=150, y=0) 
    output_text1.insert(tk.END, "Linear Regression Summary:\n")
    output_text1.insert(tk.END, linear_model.summary())
    output_text1.insert(tk.END, "\n\nPoisson Regression Summary:\n")
    output_text1.insert(tk.END, poisson_model.summary())

    canvas = FigureCanvasTkAgg(fig, master=frame)
    canvas.draw()
    canvas.get_tk_widget().place(x=150, y=240)
    
    Button(frame, width=10, pady=7, text='Back', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=search).place(x=900,y=500)
#poisson end----------------------------------
#other analysis------------------------------------------------
global df1, X_train_pca, X_test_pca, y_train, y_test
df1 = None

def append_text(output_widget, text):
    output_widget.insert(tk.END, text + "\n")
    output_widget.see(tk.END)

def load_data():
    global df1
    df1 = pd.read_excel(r"D:\PA project\survey.xlsx", sheet_name='Sheet1')
        
    
        

def handle_missing_values():
    global df1
    if df1 is not None:
        numeric_columns = df1.select_dtypes(include=[float, int]).columns
        numeric_columns_with_data = df1[numeric_columns].columns[df1[numeric_columns].notnull().any()]
        imputer = SimpleImputer(strategy='mean')
        df1[numeric_columns_with_data] = imputer.fit_transform(df1[numeric_columns_with_data])
        append_text(output_text5, "Missing values handled successfully")
    else:
        messagebox.showwarning("Warning", "Please load the data first")

def explore_data():
    global df1
    if df1 is not None:
        output = (
            f"Columns: {df1.columns}\n"
            f"First few rows:\n{df1.head()}\n"
            f"Data types:\n{df1.dtypes}\n"
            f"Number of records: {len(df1)}\n"
            f"Value counts - Gender:\n{df1['gender'].value_counts()}\n"
            f"Value counts - Language:\n{df1['language'].value_counts()}\n"
            f"Value counts - Pass/Fail:\n{df1['Pass/Fail'].value_counts()}\n"
        )
        append_text(output_text5, output)

        numeric_df1 = df1.select_dtypes(include=[float, int])
        if not numeric_df1.empty:
            sns.heatmap(numeric_df1.corr(), annot=True, cmap='coolwarm')
            plt.show()
        else:
            append_text(output_text5, "No numeric columns found for correlation computation.")
    else:
        messagebox.showwarning("Warning", "Please load the data first")

def preprocess_data():
    global df1, X_train_pca, X_test_pca, y_train, y_test
    if df1 is not None:
        df1 = pd.get_dummies(df1, columns=['gender', 'language'], drop_first=True)
        features = ['age', 'q1s', 'q2s', 'q3s', 'q4s', 'q5s', 'q6s', 'q7s', 'q8s', 'q9s', 'q10s', 'gender_male']
        if 'language_other' in df1.columns:
            features.append('language_other')
        X = df1[features]
        y = df1['Pass/Fail']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        pca = PCA(n_components=2)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_test_pca = pca.transform(X_test_scaled)
        append_text(output_text5, "Data preprocessed successfully")
    else:
        messagebox.showwarning("Warning", "Please load the data first")
def logout():
    for widget in root.winfo_children():
        widget.destroy()
  # Destroy the current window
    create_login_screen()  # Open a new login window

def open_login_window():
    # Re-create the root window for login
    root = Tk()
    root.title('Login')
    root.geometry('925x500+700+200')
    root.configure(bg="white")
    root.resizable(False, False)

    # Rest of your login window code...

    root.mainloop()
def train_models():
    global y_pred_log_reg, y_pred_tree, y_pred_forest
    if df1 is not None:
        log_reg = LogisticRegression()
        log_reg.fit(X_train_pca, y_train)
        y_pred_log_reg = log_reg.predict(X_test_pca)

        tree = DecisionTreeClassifier()
        tree.fit(X_train_pca, y_train)
        y_pred_tree = tree.predict(X_test_pca)

        forest = RandomForestClassifier()
        forest.fit(X_train_pca, y_train)
        y_pred_forest = forest.predict(X_test_pca)
        append_text(output_text5, "Models trained successfully")
    else:
        messagebox.showwarning("Warning", "Please load and preprocess the data first")

def evaluate_models():
    if df1 is not None:
        output_log_reg = (
            f"Logistic Regression:\n"
            f"{confusion_matrix(y_test, y_pred_log_reg)}\n"
            f"{classification_report(y_test, y_pred_log_reg)}\n"
        )

        output_tree = (
            f"Decision Tree Classifier:\n"
            f"{confusion_matrix(y_test, y_pred_tree)}\n"
            f"{classification_report(y_test, y_pred_tree)}\n"
        )

        output_forest = (
            f"Random Forest Classifier:\n"
            f"{confusion_matrix(y_test, y_pred_forest)}\n"
            f"{classification_report(y_test, y_pred_forest)}\n"
        )

        append_text(output_text5, output_log_reg)
        append_text(output_text5, output_tree)
        append_text(output_text5, output_forest)
    else:
        messagebox.showwarning("Warning", "Please load and preprocess the data first")

def plot_roc_curve():
    if df1 is not None:
        log_reg = LogisticRegression()
        log_reg.fit(X_train_pca, y_train)
        y_prob_log_reg = log_reg.predict_proba(X_test_pca)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob_log_reg)
        plt.plot(fpr, tpr, linestyle='-', label='Logistic Regression')
        plt.plot([0, 1], [0, 1], linestyle='--', color='r')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.show()
        auc_score = roc_auc_score(y_test, y_prob_log_reg)
        append_text(output_text5, f'AUC: {auc_score}')
    else:
        messagebox.showwarning("Warning", "Please load and preprocess the data first")

def perform_cross_validation():
    if df1 is not None:
        X = df1.drop('Pass/Fail', axis=1)
        y = df1['Pass/Fail']
        numeric_df1 = X.select_dtypes(include=[float, int])
        if not numeric_df1.empty:
            forest = RandomForestClassifier()
            cv_scores = cross_val_score(forest, numeric_df1, y, cv=5)
            output_cv = (
                f'Random Forest CV Scores: {cv_scores}\n'
                f'Random Forest Mean CV Score: {cv_scores.mean()}\n'
            )
            append_text(output_text5, output_cv)
        else:
            append_text(output_text5, "No numeric columns found for cross-validation.")
    else:
        messagebox.showwarning("Warning", "Please load and preprocess the data first")

def visualize_pca():
    if df1 is not None:
        pca_df1 = pd.DataFrame(X_train_pca, columns=['PC1', 'PC2'])
        pca_df1['Pass/Fail'] = y_train.values
        sns.scatterplot(data=pca_df1, x='PC1', y='PC2', hue='Pass/Fail', palette='Set1')
        plt.title('PCA Graph for Pass and Fail Classes')
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.show()
    else:
        messagebox.showwarning("Warning", "Please load and preprocess the data first")

def run_full_analysis():
    try:
        load_data()
        handle_missing_values()
        explore_data()
        preprocess_data()
        train_models()
        evaluate_models()
        plot_roc_curve()
        perform_cross_validation()
        visualize_pca()
    except Exception as e: 
        append_text(output_text5, f"An error occurred: {e}")

def other():
    global output_text5
    w5 = Frame(root)
    w5 = Frame(root, width=2000, height=500, bg='white')
    w5.place(x=0, y=0)
    frame = Frame(w5, width=42000, height=4500, bg='white').place(x=-100, y=-100)
    output_text5 = scrolledtext.ScrolledText(frame, width=100, height=15)
    output_text5.place(x=0,y=10)
    Button(frame, width=10, pady=7, text='Next', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=run_full_analysis).place(x=500,y=450)
    Button(frame, width=10, pady=7, text='Back', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=search).place(x=600,y=450)
#other analysis---------------------end
#student Analysis-------------------------------------
def ji():
    import sing
def stdu():
    w6 = Frame(root)
    w6 = Frame(root, width=2000, height=500, bg='white')
    w6.place(x=0, y=0)
    frame = Frame(w6, width=42000, height=4500, bg='white').place(x=-100, y=-100)
    Button(frame, width=20, pady=7, text='Check my analysis ', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=ji).place(x=400,y=450)

    Button(frame, width=10, pady=7, text='Back', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=search).place(x=600,y=450)

#clustering-------------------------------------
from sklearn.impute import SimpleImputer
def clu():
    # Clear previous contents of w3 if any
    
    # Define w3 frame
    w3=Frame(root)
    w3 = Frame(root, width=2000, height=500, bg='pink')
    w3.place(x=0, y=0)

    # Define frame inside w3
    frame = Frame(w3, width=42000, height=4500, bg='pink')
    frame.place(x=-100, y=-100)

    # Read data
    df = pd.read_excel('D:\\PA project\\app_info.xlsx')

    # Preprocess data
    df=df.drop_duplicates()
    df['size'] = df['size'].apply(lambda x: float(x[:-1]) if isinstance(x, str) and x[-1] == 'M' else None)
    df['size'].fillna(df['size'].mean(), inplace=True)
    df.dropna(inplace=True)
    
    # Perform clustering
    X = df[['minInstalls', 'score', 'ratings', 'reviews', 'size']].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    imputer = SimpleImputer(strategy='mean')
    X_scaled_imputed = imputer.fit_transform(X_scaled)
    k=3
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['cluster'] = kmeans.fit_predict(X_scaled_imputed)

    # Plotting
    fig, ax = plt.subplots(figsize=(9, 4))

    for cluster in df['cluster'].unique():
        ax.scatter(df[df['cluster'] == cluster]['minInstalls'], df[df['cluster'] == cluster]['ratings'], label=f'Cluster {cluster}')

    ax.set_xlabel('Minimum Installs')
    ax.set_ylabel('Ratings')
    ax.set_title('Clustering of Education Apps')
    ax.legend()
    Button(frame, width=10, pady=7, text='Back ', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=search).place(x=500, y=550)
    values_str = ""
    for cluster in range(3):  # Assuming 3 clusters
        cluster_data = df[df['cluster'] == cluster]
        values_str += f"Cluster {cluster}:\n"
        values_str += f"{cluster_data}\n\n"

    messagebox.showinfo("Cluster Values", values_str)
    # Embedding plot into Tkinter
    canvas = FigureCanvasTkAgg(fig, master=w3)
    canvas.draw()
    canvas.get_tk_widget().place(x=10,y=10)

#clustering end-------------------------------------------
#R^2 and MSE------------------------
def R2MSE():
    # Clear previous contents of w4 if any
    
    
    # Define w4 frame
    w4 = Frame(root, width=2000, height=500, bg='pink')
    w4.place(x=0, y=0)
    
    # Create scrolled text widget
    output_text3 = scrolledtext.ScrolledText(w4, width=100, height=10)
    output_text3.place(x=10, y=0)

    file_path = r'D:\PA project\appreview.xlsx'
    data = pd.read_excel(file_path)

    data_encoded = pd.get_dummies(data, columns=['Country/Region'], drop_first=True)

    X = data_encoded.drop(columns=['Satisfied'])
    y = data_encoded['Satisfied']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Display metrics in GUI
    output_text3.insert(tk.END, "Mean Squared Error: {:.2f}\n".format(mse))
    output_text3.insert(tk.END, "R^2 Score: {:.2f}\n".format(r2))

    predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

    different_predictions_df = predictions_df[predictions_df['Actual'] != predictions_df['Predicted']]

    # Display different predictions in GUI
    output_text3.insert(tk.END, "\nDifferent Predictions:\n{}\n".format(different_predictions_df))

    # Plot the regression graph
    fig = plt.figure(figsize=(7,3.5))  # Adjust the figure size as per your requirement
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual Satisfied")
    plt.ylabel("Predicted Satisfied")
    plt.title("Actual vs Predicted Satisfied")


    canvas = FigureCanvasTkAgg(fig, master=w4)
    canvas.draw()
    canvas.get_tk_widget().place(x=5, y=150)

    # Back button
    Button(w4, width=10, pady=7, text='Back ', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=search).place(x=750, y=400)
#R^2MSE-------------------------------------------------------end

#Factor analysis----------------------------------------------
class FactorAnalysisApp:
    def __init__(self, root):
        self.root = root
        
        self.file_path = r"D:\PA project\survey.xlsx" 

        self.create_widgets()
        self.analyze_data()

    def create_widgets(self):
        self.output_text = tk.Text(self.root, height=10, width=200)
        self.output_text.pack(side=tk.TOP, padx=10, pady=125)
        self.back_button = tk.Button(self.root, text="Back", command=search)
        self.back_button.pack(side=tk.TOP, pady=10)
        self.plot_button = tk.Button(self.root, text="show plot", command=hi)
        self.plot_button.pack(side=tk.TOP, pady=10)

        


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
            
        finally:
            print("hi")

    def display_factor_loadings(self, loadings_df):
        self.output_text.insert(tk.END, "Factor Loadings:\n")
        self.output_text.insert(tk.END, loadings_df.to_string())


reviewpredection_imported = False

def hi():
    global reviewpredection_imported

    if not reviewpredection_imported:
        import Reviewpredection
        reviewpredection_imported = True


def fac():
    global w7  # Make w7 a global variable to access it outside the function
    w7 = Frame(root)
    w7 = Frame(root, width=2000, height=500, bg='grey')
    w7.place(x=0, y=0)
    
    
    app = FactorAnalysisApp(w7)
    


#Factor analysis-------------------------------------------end
    
def back_to_a1(frame):
    frame.destroy()
    he()
def aboutus():
    ab = Frame(root)
    ab = Frame(root, width=2000, height=500, bg='brown')
    ab.place(x=0, y=0)
    
    heading = Label(ab, text='About Us\nWelcome to our E-learning Prediction platform!\n We are a team of passionate data scientists and educators\n dedicated to revolutionizing the field of online \neducation through predictive analytics. Our goal is to \nempower educators and learners with insights that enhance\n the online learning experience.', fg='black', bg='brown', font=('Times New Roman', 23, 'bold'))
    heading.place(x=80, y=5)
    Button(ab, width=29, pady=7, text='Back', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=search).place(x=350, y=350)
    root.mainloop()

def search():
    
    wun = Frame(root)
    wun = Frame(root, width=2000, height=500, bg='green')
    wun.place(x=0, y=0)
    
    
    try:
        # Load and display the background image
        img_bg = Image.open('bio.png')
        img_bg = img_bg.resize((1800, 900))
        img_bg = ImageTk.PhotoImage(img_bg)
        Label(wun, image=img_bg, bg='green').place(x=0, y=0)
    except Exception as e:
        print("Error loading background image:", e)
    
    try:
        # Load and display the second image
        img = Image.open('bio.png')
        img = img.resize((1800, 900))
        img = ImageTk.PhotoImage(img)
        Label(wun, image=img, bg='green').place(x=0, y=100)
    except Exception as e:
        print("Error loading second image:", e)
   # root.mainloop()
    
    frame = Frame(wun, width=42000, height=4500, bg='grey').place(x=-100, y=-100)
    Button(frame, width=25, pady=7, text='Linear and logistic regression', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=he).place(x=80, y=0)
    Button(frame, width=15, pady=7, text='Poisson regression', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=pos).place(x=265, y=00)
    Button(frame, width=30, pady=7, text='Clustering for the E learning App', bg='green', font=('Times New Roman', 10, 'bold'), fg='white',command=clu).place(x=380, y=0)
    Button(frame, width=19, pady=7, text='Student result analysis ', bg='green', font=('Times New Roman', 10, 'bold'), fg='white',command=stdu).place(x=600, y=0)
    Button(frame, width=10, pady=7, text='R^2 and MSE', bg='green', font=('Times New Roman', 10, 'bold'), fg='white',command=R2MSE).place(x=0, y=450)
    Button(frame, width=15, pady=7, text='Other analysis ', bg='green', font=('Times New Roman', 10, 'bold'), fg='white',command=other).place(x=81, y=450)
    Button(frame, width=10, pady=7, text='Profile', bg='green', font=('Times New Roman', 10, 'bold'), fg='white',command=pro).place(x=0, y=0)
    Button(frame, width=15, pady=7, text='Factor analysis', bg='green', font=('Times New Roman', 10, 'bold'), fg='white',command=fac).place(x=743, y=0)
    Button(frame, width=10, pady=7, text='Log out ', bg='green', font=('Times New Roman', 10, 'bold'), fg='white',command=logout).place(x=850, y=0)
    Button(frame, width=10, pady=7, text='About us', bg='green', font=('Times New Roman', 10, 'bold'), fg='white',command=aboutus).place(x=195, y=450)
    img = PhotoImage(file=r"D:\PA project\bio.png",width=1000,height=700)
    Label(wun,width=1000,height=700,image=img,bg='brown').place(x=0,y=0)
    root.mainloop()
def he():
    global a1
    global output_text
    a1 = Frame(root)
    a1 = Frame(root, width=2000, height=500, bg='white')
    a1.place(x=0, y=0)

    frame = Frame(a1, width=42000, height=4500, bg='white').place(x=-100, y=-100)
    try:
        # Load and display the background image
        img_bg = Image.open('eng2.png')
        img_bg = img_bg.resize((1800, 900))
        img_bg = ImageTk.PhotoImage(img_bg)
        Label(a1, image=img_bg, bg='green').place(x=0, y=0)
    except Exception as e:
        print("Error loading background image:", e)
    
    try:
        # Load and display the second image
        img = Image.open('eng2.png')
        img = img.resize((1800, 900))
        img = ImageTk.PhotoImage(img)
        Label(a1, image=img, bg='green').place(x=0, y=100)
    except Exception as e:
        print("Error loading second image:", e)
    
    
    plt.figure(figsize=(10, 6))
    Button(frame, width=30, pady=7, text='Analyze', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=analyze_regression).place(x=360, y=270)

    Button(frame, width=30, pady=7, text='Show the analyzed data', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=regal).place(x=360, y=320)
   
    Button(frame, width=30, pady=7, text='Back ', bg='green', font=('Times New Roman', 10, 'bold'), fg='white', command=search).place(x=360, y=370)
    img = PhotoImage(file=r"D:\PA project\eng2.png",width=1000,height=700)
    Label(a1,width=1000,height=700,image=img,bg='brown').place(x=0,y=0)
    root.mainloop()
    root.mainloop()
def login():
    global logged_in_user
    
    

    username = user.get()
    password = code.get()
    gender = gen.get()
    
            
    if not username or not password or not gender:
        login_status.config(text="Username, password, and gender cannot be empty")
        return

    try:
       
        cursor.execute("SELECT * FROM user WHERE username=%s AND password=%s AND gender=%s", (username, password, gender))
        user_data = cursor.fetchone()

        if user_data:
            logged_in_user = user_data
            login_status.config(text="Login successful!")
            search()
        else:
            login_status.config(text="Invalid username, password, or gender")
    except mysql.connector.Error as err:
        login_status.config(text=f"Error: {err}")

def signup():
    username = user.get()
    password = code.get()
    gender = gen.get()

    if not username or not password or not gender:
        signup_status.config(text="Username, password, and gender cannot be empty")
        return

    try:
        cursor.execute("SELECT * FROM user WHERE username=%s", (username,))
        existing_user = cursor.fetchone()

        if existing_user:
            signup_status.config(text="Username already exists")
        else:
            cursor.execute("INSERT INTO user (username, password, gender) VALUES (%s, %s, %s)", (username, password, gender))
            connection.commit()
            signup_status.config(text="Signup successful!")
            
    except mysql.connector.Error as err:
        signup_status.config(text=f"Error: {err}")

db_url = "localhost"
username = "root"
password = "gugan"
database = "hi"

connection = mysql.connector.connect(host=db_url, user=username, password=password, database=database)

if connection.is_connected():
    print("Connected to the database")
    cursor = connection.cursor()
else:
    print("Failed to connect to the database")

root = Tk()
root.title('login')
root.geometry('925x500+700+200')
root.configure(bg="white")
root.resizable(False, False)

def fp_on_enter(e):
    fp_user.delete(0, 'end')

def fp_on_leave(e):
    name = fp_user.get()
    if name == '':
        fp_user.insert(0, 'Username')
#user end-------------------------
#gender---------------------------
def fp_on_enter_gen(e):
    fp_gen.delete(0, 'end')

def fp_on_leave_gen(e):
    name = fp_gen.get()
    if name == '':
        fp_gen.insert(0, 'Gender')
#gen----------------------------
def fp_on_enter_code(e):
    fp_code.delete(0, 'end')

def fp_on_leave_code(e):
    name = fp_code.get()
    if name == '':
        fp_code.insert(0, 'Password')
def on_enter(e):
    user.delete(0, 'end')

def on_leave(e):
    name = user.get()
    if name == '':
        user.insert(0, 'Username')
#user end-------------------------
#gender---------------------------
def on_enter_gen(e):
    gen.delete(0, 'end')

def on_leave_gen(e):
    name = gen.get()
    if name == '':
        gen.insert(0, 'Gender')
#gen----------------------------
def on_enter_code(e):
    code.delete(0, 'end')

def on_leave_code(e):
    name = code.get()
    if name == '':
        code.insert(0, 'Password')
def submit_new_password():
      username = fp_user.get()
      gender = fp_gen.get()
      new_password = fp_code.get()
      if not username or not gender or not new_password:
          fp_status.config(text="All fields are required")
          return

      try:
          cursor.execute("SELECT * FROM user WHERE username=%s AND gender=%s", (username, gender))
          user_data = cursor.fetchone()

          if user_data:
               cursor.execute("UPDATE user SET password=%s WHERE username=%s AND gender=%s", (new_password, username, gender))
               connection.commit()
               fp_status.config(text="Password reset successful!")
          else:
               fp_status.config(text="Invalid username or gender")
      except mysql.connector.Error as err:
          fp_status.config(text=f"Error: {err}")
def fpass():
    global fp_user,fp_code,fp_gen,fp_status
    for widget in root.winfo_children():
        widget.destroy()

    fp1 = Frame(root, width=2000, height=500, bg='white')
    fp1.place(x=0, y=0)
    heading = Label(fp1, text='Your info', fg='black', bg='white', font=('Times New Roman', 23, 'bold'))
    heading.place(x=450, y=5)
    fp_user = Entry(fp1, width=30, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    fp_user.place(x=350, y=80)
    fp_user.insert(0, 'Username')
    fp_user.bind('<FocusIn>', fp_on_enter)
    fp_user.bind('<FocusOut>', fp_on_leave)

    
    fp_code = Entry(fp1, width=30, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    fp_code.place(x=350, y=135 )
    fp_code.insert(0, 'Password')
    fp_code.bind('<FocusIn>', fp_on_enter_code)
    fp_code.bind('<FocusOut>', fp_on_leave_code)
    
    fp_gen = Entry(fp1, width=30, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    fp_gen.place(x=350, y=190)
    fp_gen.insert(0, 'Gender')
    fp_gen.bind('<FocusIn>', fp_on_enter_gen)
    fp_gen.bind('<FocusOut>',fp_on_leave_gen)
    
    Button(fp1, width=29, pady=7, text='Change password ', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3,command=submit_new_password).place(x=350, y=300)
    fp_status = Label(fp1, text='', fg='red', bg='white', font=('Times New Roman', 10, 'bold'))
    fp_status.place(x=350, y=350)
    Button(fp1, width=29, pady=7, text='Back to Log in', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=create_login_screen).place(x=350, y=250)

    
def create_login_screen():
    
    global user, code,gen, login_status, signup_status
    for widget in root.winfo_children():
        widget.destroy()
    
    # Display the image
    img = PhotoImage(file='bg.png',width=1000,height=700)
    Label(root,width=1000,height=700,image=img).place(x=0,y=0)
    frame=Frame(root,width=320,height=450,bg='brown')
    frame.place(x=480,y=70)
    
    heading=Label(frame,text='Sign in',fg='black',bg='brown',font=('Times New Roman',23,'bold'))
    heading.place(x=100,y=5)
    user = Entry(frame, width=20, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    user.place(x=30, y=80)
    user.insert(0, 'Username')
    user.bind('<FocusIn>', on_enter)
    user.bind('<FocusOut>', on_leave)

    code = Entry(frame, width=20, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    code.place(x=30, y=135)
    code.insert(0, 'Password')
    code.bind('<FocusIn>', on_enter_code)
    code.bind('<FocusOut>', on_leave_code)
    
    gen = Entry(frame, width=20, fg='black', border=2, bg='lightblue', font=('Times New Roman', 14, 'bold'))
    gen.place(x=30, y=190)
    gen.insert(0, 'Gender')
    gen.bind('<FocusIn>', on_enter_gen)
    gen.bind('<FocusOut>', on_leave_gen)
    Frame(frame, width=295, height=2, bg='black').place(x=25, y=107)
    Frame(frame, width=295, height=2, bg='black').place(x=25, y=163)
    Frame(frame, width=295, height=2, bg='black').place(x=25, y=218)
    Button(frame, width=15, pady=7, text='Log in', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=login).place(x=50, y=250)
    
    Button(frame, width=15, pady=7, text='Sign up', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3, command=signup).place(x=170, y=250)
    Button(frame, width=15, pady=7, text='Forget password', bg='#57a1f8', font=('Times New Roman', 10, 'bold'), fg='black', border=3,command=fpass).place(x=120, y=300)

    login_status = Label(frame, text='', fg='red', bg='brown', font=('Times New Roman', 10, 'bold'))
    login_status.place(x=30, y=280)

    signup_status = Label(frame, text='', fg='red', bg='brown', font=('Times New Roman', 10, 'bold'))
    signup_status.place(x=30, y=330)
    root.mainloop()
   

create_login_screen()

