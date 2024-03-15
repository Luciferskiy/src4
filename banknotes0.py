import csv 
import random

from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeightborsClassifier

# model = Perceptron()
# model = svm.SVC()
# model = KNeighborsClassifier(n_neighbors=1)
model = GaussianNB()

# Read data in form file
with open("banknotes.csv") as f:
  reader = csv.reader(f)
  next(reader)

  data = []
  for row in reader:
    data.append({
      "evidence": [float(cell) for cell in row[:4]],
      "label": "Authentic" if row [4] == "0" else "Counterfelt"
    })

# Separate data into training and testing groups
holdout = int(0.50 * len(data))
random.shuffle(data)
testing = data[:holdout]
training = data[holdout:]

# Train model on training set
X_training = [row["evidence"] for row in training]
y_training = [row["label"] for row in training]
model.fit(X_training, y_training_)

# Make predictions on the testing set
X_training = [row["evidence"] for row in training]
y_testing = [row["label"] for row in testing]
predictions = model.predict(X_testing)


# Compute how well we performed
correct = 0
incorrect = 0
total = 0 
for actual, predicted in zip(y_testing, predictions):
  total += 1
  if actual == predicted:
    correct += 1
  else:
    incorrect += 1

# Print results
print(f"Results for model{type(model).__name__}")
print(f"Correct: {correct}")
print(f"Incorrect: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")
