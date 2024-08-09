#Amirhossein shanaghi 810899056
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import numpy as np

boston = load_boston()
X, y = boston.data, boston.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


sgd_regressor = SGDRegressor(penalty='none', max_iter=1000, tol=1e-3, learning_rate='invscaling', eta0=0.01)

sgd_regressor.fit(X_train_scaled, y_train)

y_pred_train_sgd = sgd_regressor.predict(X_train_scaled)
y_pred_test_sgd = sgd_regressor.predict(X_test_scaled)

mse_train_sgd = np.mean((y_train - y_pred_train_sgd) ** 2)
mse_test_sgd = np.mean((y_test - y_pred_test_sgd) ** 2)

weights_sgd = sgd_regressor.coef_
bias_sgd = sgd_regressor.intercept_

mse_train_sgd, mse_test_sgd, weights_sgd, bias_sgd
