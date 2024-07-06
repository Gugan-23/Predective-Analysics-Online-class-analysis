import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

df = pd.read_excel('D:\\PA project\\app_info.xlsx')

df['size'] = df['size'].apply(lambda x: float(x[:-1]) if x[-1] == 'M' else None)

df['size'].fillna(df['size'].mean(), inplace=True)

X = df[['minInstalls', 'score', 'ratings', 'reviews', 'size']].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

k = 3
kmeans = KMeans(n_clusters=k, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

root = tk.Tk()
root.title("Clustering of Education Apps")

fig, ax = plt.subplots(figsize=(10, 6))

for cluster in df['cluster'].unique():
    ax.scatter(df[df['cluster'] == cluster]['minInstalls'], df[df['cluster'] == cluster]['ratings'], label=f'Cluster {cluster}')
ax.set_xlabel('Minimum Installs')
ax.set_ylabel('Ratings')
ax.set_title('Clustering of Education Apps')
ax.legend()


canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

text = tk.Text(root)
text.insert(tk.END, df[['name', 'cluster']])
text.pack()

root.mainloop()
