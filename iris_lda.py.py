import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import datasets
import matplotlib

matplotlib.use('TkAgg')
iris = datasets.load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X, y)
df_lda = pd.DataFrame(X_lda, columns=['LD1', 'LD2'])
df_lda['Target'] = y
plt.figure(figsize=(8, 6))
colors = ['r', 'g', 'b']
for i, target_name in enumerate(target_names):
    plt.scatter(df_lda.loc[df_lda['Target'] == i, 'LD1'],
                df_lda.loc[df_lda['Target'] == i, 'LD2'],
                label=target_name, color=colors[i], alpha=0.7)
plt.xlabel("Linear Discriminant 1")
plt.ylabel("Linear Discriminant 2")
plt.title("LDA of Iris Dataset")
plt.legend()
plt.grid()
plt.show()
