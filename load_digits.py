from sklearn.datasets import load_digits
import matplotlib.pyplot as plt


dataset = load_digits()

print(dataset.data)
print(dataset.target)
print(dataset.data.shape)
print(dataset.images.shape)

datalength = len(dataset.images)
print(datalength)
n = 7
plt.gray()
plt.matshow(dataset.images[n])
plt.show()

X = dataset.images.reshape((len(dataset.images), -1))
y = dataset.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

print(X_train.shape)
print(X_test.shape)

from sklearn import svm
model = svm.SVC(kernel='linear')

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

acc = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(acc * 100))

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

print("Classification Report:")
print(classification_report(y_test, y_pred))

n = 118
plt.matshow(dataset.images[n])
plt.show()
