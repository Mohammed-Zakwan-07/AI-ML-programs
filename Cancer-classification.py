import os
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

os.chdir("C:/Users/hi/Downloads")
df = pd.read_csv("breast-cancer.csv")
df = df.drop('id', axis=1)
y = df['diagnosis'].map({'B':0, 'M':1})
X = df.drop('diagnosis', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic regression model------------------------------------------------------
modellogistic = LogisticRegression(max_iter=10000)
modellogistic.fit(X_train, y_train)
y_pred_logistic = modellogistic.predict(X_test)
cm_logistic = confusion_matrix(y_test, y_pred_logistic)
classification_report_logistic = classification_report(y_test, y_pred_logistic)
accuracy_logistic = accuracy_score(y_test ,y_pred_logistic)
print("---------------Logistic regression model--------------------------------\n")
print("Confusion Matrix : \n",cm_logistic)
print("Classification Report : \n", classification_report_logistic)
print("Accuracy : ", accuracy_logistic)
print("------------------------------------------------------------------------")




# XGB Boost -----------------------------------------------------------------------
modelXGBoost = XGBClassifier()
modelXGBoost.fit(X_train, y_train)
y_pred_XGBoost = modelXGBoost.predict(X_test)
cm_XGBoost = confusion_matrix(y_test, y_pred_XGBoost)
classification_report_XGBoost = classification_report(y_test, y_pred_XGBoost)
accuracy_XGBoost = accuracy_score(y_test ,y_pred_XGBoost)
print("---------------XGBoost Model--------------------------------------------\n")
print("Confusion Matrix : \n",cm_XGBoost)
print("Classification Report : \n", classification_report_XGBoost)
print("Accuracy : ", accuracy_XGBoost)
print("------------------------------------------------------------------------")



#Decision Tree------------------------------------------------------------------
model_DecisionTreeClassifier = DecisionTreeClassifier(
    max_depth=5,          # limits overfitting
    min_samples_split=10, # avoid tiny splits
    random_state=42
)
model_DecisionTreeClassifier.fit(X_train, y_train)
y_pred_DecisionTreeClassifier = model_DecisionTreeClassifier.predict(X_test)
cm_DecisionTreeClassifier = confusion_matrix(y_test, y_pred_DecisionTreeClassifier)
classification_report_DecisionTreeClassifier = classification_report(y_test, y_pred_DecisionTreeClassifier)
accuracy_score_DecisionTreeClassifier = accuracy_score(y_test, y_pred_DecisionTreeClassifier)
print("---------------DecisionTreeClassifier Model-----------------------------\n")
print("Confusion Matrix : \n",cm_DecisionTreeClassifier)
print("Classification Report : \n", classification_report_DecisionTreeClassifier)
print("Accuracy : ", accuracy_score_DecisionTreeClassifier)
print("------------------------------------------------------------------------")


# KNN ----------------------------------------------------------------------------
modelKNN = KNeighborsClassifier(n_neighbors=5)
modelKNN.fit(X_train, y_train)
y_pred_KNN = modelKNN.predict(X_test)
accuracy_KNN = accuracy_score(y_test, y_pred_KNN)
print("---------------KNN Model------------------------------------------------\n")
print("Confusion Matrix : \n", confusion_matrix(y_test, y_pred_KNN))
print("Classification Report : \n", classification_report(y_test, y_pred_KNN))
print("Accuracy : ", accuracy_KNN)
print("------------------------------------------------------------------------")

# Comparison Bar Chart -----------------------------------------------------------
models = ['Logistic Regression', 'XGBoost', 'Decision Tree', 'KNN']
accuracies = [accuracy_logistic, accuracy_XGBoost, accuracy_score_DecisionTreeClassifier, accuracy_KNN]

plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red'])
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0.85, 1.02)
plt.show()

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'XGBoost', 'Decision Tree', 'KNN'],
    'Accuracy': [accuracy_logistic, accuracy_XGBoost, accuracy_score_DecisionTreeClassifier, accuracy_KNN]
})
results = results.sort_values('Accuracy', ascending=False)
print(results.to_string(index=False))