import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

os.chdir('C:/Users/hi/Downloads')
df = pd.read_csv("Computer_Data.csv")

df = df.drop(df.columns[df.columns.str.contains('unnamed', case=False)], axis=1)


df['cd'] = df['cd'].map({'yes': 1, 'no': 0})
df['multi'] = df['multi'].map({'yes': 1, 'no': 0})
df['premium'] = df['premium'].map({'yes': 1, 'no': 0})

X = df.drop('price', axis=1)
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


print("R2 Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
