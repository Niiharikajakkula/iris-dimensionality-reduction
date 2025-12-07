import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris
import matplotlib
matplotlib.use('TkAgg')
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
tsne = TSNE(n_components=2, random_state=50)
X_tsne = tsne.fit_transform(X_scaled)
tsne_df = pd.DataFrame(X_tsne, columns=['TSNE1', 'TSNE2'])
tsne_df['species'] = y
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i, target_name in enumerate(target_names):
    plt.scatter(tsne_df[tsne_df['species'] == i]['TSNE1'],
                tsne_df[tsne_df['species'] == i]['TSNE2'],
                label=target_name, color=colors[i])
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Iris Dataset')
plt.legend()
plt.grid()
plt.show()
