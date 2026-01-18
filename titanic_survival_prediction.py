import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression

os.chdir("C:/Users/hi/Downloads")
df = pd.read_csv("titanic.csv")

df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()

df["Sex"] = sex_encoder.fit_transform(df["Sex"])
df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])

X = df.drop("Survived", axis=1)
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

sample = pd.DataFrame({
    "Pclass": [1],
    "Sex": sex_encoder.transform(["female"]),
    "Age": [28],
    "SibSp": [0],
    "Parch": [0],
    "Fare": [120],
    "Embarked": embarked_encoder.transform(["C"])
})


print("Sample prediction:", model.predict(sample))
