import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# 1. Load dataset
os.chdir("C:/Users/hi/Downloads")
df = pd.read_csv("titanic.csv")

# 2. Select useful columns
df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]

# 3. handle missing values safely
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])


# 4. Separate encoders
sex_encoder = LabelEncoder()
embarked_encoder = LabelEncoder()

df["Sex"] = sex_encoder.fit_transform(df["Sex"])
df["Embarked"] = embarked_encoder.fit_transform(df["Embarked"])

# 5. Split features and labels
X = df.drop("Survived", axis=1)
y = df["Survived"]

# 6. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 7. Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 8. Predict
y_pred = model.predict(X_test)

# 9. Evaluate
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 10. Predict new data
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
