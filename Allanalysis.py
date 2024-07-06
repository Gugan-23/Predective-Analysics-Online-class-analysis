import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

# Global variables
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
        append_text(output_text, "Missing values handled successfully")
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
        append_text(output_text, output)

        numeric_df1 = df1.select_dtypes(include=[float, int])
        if not numeric_df1.empty:
            sns.heatmap(numeric_df1.corr(), annot=True, cmap='coolwarm')
            plt.show()
        else:
            append_text(output_text, "No numeric columns found for correlation computation.")
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
        append_text(output_text, "Data preprocessed successfully")
    else:
        messagebox.showwarning("Warning", "Please load the data first")

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
        append_text(output_text, "Models trained successfully")
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

        append_text(output_text, output_log_reg)
        append_text(output_text, output_tree)
        append_text(output_text, output_forest)
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
        append_text(output_text, f'AUC: {auc_score}')
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
            append_text(output_text, output_cv)
        else:
            append_text(output_text, "No numeric columns found for cross-validation.")
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
        append_text(output_text, f"An error occurred: {e}")

# GUI setup
root = tk.Tk()
root.title("Data Analysis GUI")

analyze_button = tk.Button(root, text="Run Full Analysis", command=run_full_analysis)
analyze_button.pack(pady=5)

# Text widget for output
output_text = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=20)
output_text.pack(pady=10)

root.mainloop()
