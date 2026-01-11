from sklearn.datasets import load_iris
import pandas as pd

# Load Iris data
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['target'] = iris.target

print("\n--- Iris Dataset ---\n")
print(df.head())

print("\n--- Info ---")
print(df.info())

print("\n--- Summary ---\n")
print(df.describe())
