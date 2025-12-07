 import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
print(iris.keys())
print(iris.target_names)
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df)
