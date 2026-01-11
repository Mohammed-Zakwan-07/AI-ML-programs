import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
os.chdir("C:/Users/hi/Downloads")
# ----------------------------------------------
# 1. Load the dataset
# ----------------------------------------------
folder_path = r"C:\Users\hi\Downloads"
df = pd.read_csv("Exam_Score_Prediction.csv")   # change filename if needed

print("\n--- Dataset Loaded ---")
print(df.head())

# ----------------------------------------------
# 2. Basic info
# ----------------------------------------------
print("\n--- Shape (rows, columns) ---")
print(df.shape)

print("\n--- Column Names ---")
print(df.columns)

print("\n--- Info ---")
print(df.info())

# ----------------------------------------------
# 3. Check missing values
# ----------------------------------------------
print("\n--- Missing Values ---")
print(df.isnull().sum())

# Fill missing numeric values
df = df.fillna(df.mean(numeric_only=True))

# Fill missing categorical values
df = df.fillna(df.mode().iloc[0])

# ----------------------------------------------
# 4. Convert categorical columns (Label Encoding)
# ----------------------------------------------
label_encoder = LabelEncoder()

for col in df.columns:
    if df[col].dtype == "object":        # categorical column
        df[col] = label_encoder.fit_transform(df[col])

print("\n--- After Label Encoding ---")
print(df.head())

# ----------------------------------------------
# 5. Correlation Analysis
# ----------------------------------------------
print("\n--- Correlation Matrix ---")
print(df.corr())

# ----------------------------------------------
# 6. Selecting target and features
# ----------------------------------------------
# Change "Exam_Score" to your actual target column name
target_column = "exam_score"

X = df.drop(target_column, axis=1)  # features
y = df[target_column]               # target

# ----------------------------------------------
# 7. Train-Test Split
# ----------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------------------------
# 8. Train a simple ML model
# ----------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Model Trained Successfully ---")

# ----------------------------------------------
# 9. Make predictions
# ----------------------------------------------
y_pred = model.predict(X_test)

print("\n--- Predictions ---")
print(y_pred[:10])

# ----------------------------------------------
# 10. Evaluate the model
# ----------------------------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\n--- Model Evaluation ---")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# ----------------------------------------------
# 11. Feature Importance
# ----------------------------------------------
importance = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_
})

print("\n--- Feature Importance (Coefficients) ---")
print(importance.sort_values(by="Coefficient", ascending=False))
