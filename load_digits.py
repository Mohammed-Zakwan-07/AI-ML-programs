from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# 1. Load the dataset
digits = load_digits()

plt.gray()
plt.matshow(digits.images[9])
plt.title(f"Digit: {digits.target[0]}")
plt.show()

# Features (pixel values)
X = digits.data

# Labels (which digit it is)
y = digits.target
print(digits.target)

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Create and train the model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

# 5. Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# 6. Predict a single digit
print("Prediction for first test image:", model.predict([X_test[5]]))
