import os
import pandas as pd

os.chdir('C:/Users/hi/Downloads')
teledata = pd.read_csv('Telecom_data.csv')

teledata = teledata.drop(['phone number'], axis=1)

teledata['international plan'] = teledata['international plan'].map({'yes': 1, 'no': 0})
teledata['voice mail plan'] = teledata['voice mail plan'].map({'yes': 1, 'no': 0})

teledata = pd.get_dummies(teledata, columns=['state'], drop_first=True)

X = teledata.drop('churn', axis=1)
y = teledata['churn']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
print("Accuracy score = {:.2f}%".format(acc * 100))

tn, fp, fn, tp = cm.ravel()
print(tn, fp, fn, tp)