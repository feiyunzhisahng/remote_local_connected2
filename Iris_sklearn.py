from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets (120 training samples, 30 testing samples)， test_size=0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# L1 Regularization， C=1（default）
l1_model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
l1_model.fit(X_train, y_train)

# Predictions on training and testing sets using L1 model
y_train_pred_l1 = l1_model.predict(X_train)
y_test_pred_l1 = l1_model.predict(X_test)

# count non-zero coefficients of l1_model
print(l1_model.coef_)
print((l1_model.coef_ != 0).sum(axis=1))

# Calculate accuracy on training and testing sets using L1 model
train_accuracy_l1 = accuracy_score(y_train, y_train_pred_l1)
test_accuracy_l1 = accuracy_score(y_test, y_test_pred_l1)

print("L1 Regularization:")
print(f"Training Accuracy: {train_accuracy_l1:.2f}")
print(f"Testing Accuracy: {test_accuracy_l1:.2f}")

# L2 Regularization, C=1
l2_model = LogisticRegression(penalty="l2", solver="liblinear", max_iter=1000)
l2_model.fit(X_train, y_train)

# Predictions on training and testing sets using L2 model
y_train_pred_l2 = l2_model.predict(X_train)
y_test_pred_l2 = l2_model.predict(X_test)

# count non-zero coefficients of l2_model
print(l2_model.coef_)
print((l2_model.coef_ != 0).sum(axis=1))

# Calculate accuracy on training and testing sets using L2 model
train_accuracy_l2 = accuracy_score(y_train, y_train_pred_l2)
test_accuracy_l2 = accuracy_score(y_test, y_test_pred_l2)

print("\nL2 Regularization:")
print(f"Training Accuracy: {train_accuracy_l2:.2f}")
print(f"Testing Accuracy: {test_accuracy_l2:.2f}")

# search for the best i ranging from 0.05-1.00 at 0.05 each step
l1 = []
l2 = []
l1test = []
l2test = []
for i in np.linspace(0.05,1,19):
    lrl1 = LogisticRegression(penalty="l1",solver="liblinear",C=i,max_iter=1000)
    lrl2 = LogisticRegression(penalty="l2",solver="liblinear",C=i,max_iter=1000)
    lrl1 = lrl1.fit(X_train,y_train)
    l1.append(accuracy_score(lrl1.predict(X_train),y_train))
    l1test.append(accuracy_score(lrl1.predict(X_test),y_test))
    lrl2 = lrl2.fit(X_train, y_train)
    l2.append(accuracy_score(lrl2.predict(X_train), y_train))
    l2test.append(accuracy_score(lrl2.predict(X_test), y_test))

# plot
graph = [l1, l2, l1test, l2test]
color = ["green", "black", "lightgreen", "gray"]
label = ["L1", "L2", "L1test", "L2test"]

plt.figure(figsize=(6, 6))
for i in range(len(graph)):
    plt.plot(np.linspace(0.05, 1, 19), graph[i], color[i], label=label[i])
plt.legend(loc=4) 
plt.show()

