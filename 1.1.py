#Amirhossein shanaghi 810899056
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

boston = load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

class MiniBatchGradientDescent:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            for i in range(0, n_samples, self.batch_size):
                X_i = X_shuffled[i:i+self.batch_size]
                y_i = y_shuffled[i:i+self.batch_size]
                y_predicted = np.dot(X_i, self.weights) + self.bias

                # Calculate gradients
                dw = (1 / self.batch_size) * np.dot(X_i.T, (y_predicted - y_i))
                db = (1 / self.batch_size) * np.sum(y_predicted - y_i)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

mbgd = MiniBatchGradientDescent(learning_rate=0.01, n_iterations=1000, batch_size=32)
mbgd.fit(X_train_scaled, y_train)

y_pred_train = mbgd.predict(X_train_scaled)
y_pred_test = mbgd.predict(X_test_scaled)

mse_train = mbgd.mean_squared_error(y_train, y_pred_train)
mse_test = mbgd.mean_squared_error(y_test, y_pred_test)

mse_train, mse_test



class MiniBatchGradientDescentL2:
    def __init__(self, learning_rate=0.01, n_iterations=1000, batch_size=32, regularization_strength=0.1):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.batch_size = batch_size
        self.regularization_strength = regularization_strength
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            idx = np.random.permutation(n_samples)
            X_shuffled = X[idx]
            y_shuffled = y[idx]
            for i in range(0, n_samples, self.batch_size):
                X_i = X_shuffled[i:i+self.batch_size]
                y_i = y_shuffled[i:i+self.batch_size]
                y_predicted = np.dot(X_i, self.weights) + self.bias

                # Calculate gradients
                dw = (1 / self.batch_size) * np.dot(X_i.T, (y_predicted - y_i)) + (self.regularization_strength * self.weights)
                db = (1 / self.batch_size) * np.sum(y_predicted - y_i)

                # Update parameters
                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def mean_squared_error(self, y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2) + self.regularization_strength * np.sum(self.weights ** 2)

# Instantiate and train the L2 regularized model
mbgd_l2 = MiniBatchGradientDescentL2(learning_rate=0.01, n_iterations=1000, batch_size=32, regularization_strength=0.1)
mbgd_l2.fit(X_train_scaled, y_train)

# Predict and evaluate with L2 regularization
y_pred_train_l2 = mbgd_l2.predict(X_train_scaled)
y_pred_test_l2 = mbgd_l2.predict(X_test_scaled)

mse_train_l2 = mbgd_l2.mean_squared_error(y_train, y_pred_train_l2)
mse_test_l2 = mbgd_l2.mean_squared_error(y_test, y_pred_test_l2)

mse_train_l2, mse_test_l2