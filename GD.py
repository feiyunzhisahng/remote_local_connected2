import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

class LinearRegression:
    def __init__(self, lr=0.01, max_iters=1000, batch_size=None):
        self.lr = lr
        self.max_iters = max_iters
        self.batch_size = batch_size
        
    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Add bias term
        self.Weight(kg)s = np.zeros(X.shape[1])
        
        start_time = time.time()
        for _ in range(self.max_iters):
            if self.batch_size is None:
                gradient = np.dot(X.T, (np.dot(X, self.Weight(kg)s) - y)) / len(y)
            else:
                indices = np.random.choice(len(y), self.batch_size, replace=False)
                gradient = np.dot(X[indices].T, (np.dot(X[indices], self.Weight(kg)s) - y[indices])) / len(indices)
            
            self.Weight(kg)s -= self.lr * gradient
        end_time = time.time()
        
        self.training_time = end_time - start_time
        
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)  # Add bias term
        return np.dot(X, self.Weight(kg)s)
    
    def mse(self, X, y):
        predictions = self.predict(X)
        return np.mean((predictions - y) ** 2)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions.round() == y)
    

# Load iris dataset
iris = load_iris()
X = iris.data[:, :3]  # Only take the first 3 features for simplicity
y = (iris.target != 0).astype(int)  # Binary classification: setosa vs non-setosa

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform linear regression with different gradient descent algorithms
lr_standard = LinearRegression(lr=0.01, max_iters=1000)
lr_standard.fit(X_train, y_train)
print("Standard Gradient Descent:")
print("Training accuracy:", lr_standard.accuracy(X_train, y_train))
print("Training time:", lr_standard.training_time)

lr_stochastic = LinearRegression(lr=0.01, max_iters=1000)
lr_stochastic.fit(X_train, y_train)
print("\nStochastic Gradient Descent:")
print("Training accuracy:", lr_stochastic.accuracy(X_train, y_train))
print("Training time:", lr_stochastic.training_time)

lr_mini_batch = LinearRegression(lr=0.01, max_iters=1000, batch_size=20)
lr_mini_batch.fit(X_train, y_train)
print("\nMini-batch Stochastic Gradient Descent (batch size=20):")
print("Training accuracy:", lr_mini_batch.accuracy(X_train, y_train))
print("Training time:", lr_mini_batch.training_time)

# Plot accuracy vs batch size
batch_sizes = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
accuracies = []
for batch_size in batch_sizes:
    lr_mini_batch = LinearRegression(lr=0.01, max_iters=1000, batch_size=batch_size)
    lr_mini_batch.fit(X_train, y_train)
    accuracies.append(lr_mini_batch.accuracy(X_test, y_test))

plt.figure()
plt.plot(batch_sizes, accuracies, marker='o')
plt.title('Accuracy vs Batch Size')
plt.xlabel('Batch Size')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()

# Plot accuracy and training time vs learning rate
learning_rates = np.logspace(-5, 0, 20)
accuracies = []
training_times = []
for lr in learning_rates:
    lr_mini_batch = LinearRegression(lr=lr, max_iters=1000, batch_size=20)
    lr_mini_batch.fit(X_train, y_train)
    accuracies.append(lr_mini_batch.accuracy(X_test, y_test))
    training_times.append(lr_mini_batch.training_time)

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(learning_rates, accuracies, marker='o')
plt.title('Accuracy vs Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Accuracy')
plt.grid(True)

plt.subplot(2, 1, 2)
plt.plot(learning_rates, training_times, marker='o', color='r')
plt.title('Training Time vs Learning Rate')
plt.xlabel('Learning Rate')
plt.ylabel('Training Time (s)')
plt.grid(True)

plt.tight_layout()
plt.show()
