import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("D:/Downloads/Smartphone_Usage_Productivity_Dataset_50000.csv")

df = df[["Age", "Gender", "Occupation", "Device_Type",
         "Daily_Phone_Hours", "Social_Media_Hours",
         "Sleep_Hours", "App_Usage_Count",
         "Caffeine_Intake_Cups", "Weekend_Screen_Time_Hours",
         "Work_Productivity_Score"]]

df = pd.get_dummies(df, drop_first=True)

X = df.drop("Work_Productivity_Score", axis=1)
y = df["Work_Productivity_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("R2:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
