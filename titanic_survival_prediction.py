import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Set directory
os.chdir("C:/Users/hi/Downloads")

# Load dataset
df = pd.read_csv("titanic.csv")



print((df["Sex"] == 0).sum())
print(df["Age"].mode())
print(df["Fare"].sum())
print(df["Fare"].describe())
