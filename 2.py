#Amirhossein shanaghi 810899056

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(X, y, theta, lambda_=0):
    m = len(y)
    h = sigmoid(X @ theta)
    epsilon = 1e-5  # to prevent log(0)
    cost = (-1/m) * (y.T @ np.log(h + epsilon) + (1 - y).T @ np.log(1 - h + epsilon))
    reg_cost = (lambda_ / (2*m)) * np.sum(theta[1:] ** 2)
    return cost + reg_cost

def gradient_descent(X, y, theta, alpha, num_iters, lambda_=0):
    m = len(y)
    J_history = []

    for i in range(num_iters):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y)) + (lambda_ / m) * np.r_[[[0]], theta[1:].reshape(-1, 1)]
        theta = theta - alpha * gradient
        J_history.append(compute_cost(X, y, theta, lambda_))

    return theta, J_history

X_train_scaled_with_intercept = np.concatenate([np.ones((X_train_scaled.shape[0], 1)), X_train_scaled], axis=1)
X_test_scaled_with_intercept = np.concatenate([np.ones((X_test_scaled.shape[0], 1)), X_test_scaled], axis=1)

initial_theta = np.zeros((X_train_scaled_with_intercept.shape[1], 1))

# Set hyperparameters
alpha = 0.01
num_iters = 1000

# Perform gradient descent (without regularization)
theta, J_history = gradient_descent(X_train_scaled_with_intercept, y_train.reshape(-1, 1), initial_theta, alpha, num_iters)

cost_train = compute_cost(X_train_scaled_with_intercept, y_train, theta)

cost_test = compute_cost(X_test_scaled_with_intercept, y_test, theta)

cost_train, cost_test, theta.ravel()[:5], len(J_history)

# Perform gradient descent with L2 regularization (lambda = 1)
lambda_reg = 1
theta_reg, J_history_reg = gradient_descent(X_train_scaled_with_intercept, y_train.reshape(-1, 1), initial_theta, alpha, num_iters, lambda_reg)

cost_train_reg = compute_cost(X_train_scaled_with_intercept, y_train, theta_reg, lambda_reg)

cost_test_reg = compute_cost(X_test_scaled_with_intercept, y_test, theta_reg, lambda_reg)

cost_train_reg, cost_test_reg, theta_reg.ravel()[:5], len(J_history_reg)

