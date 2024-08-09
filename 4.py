#Amirhossein shanaghi 810899056

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

data = load_breast_cancer()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled.shape, X_test_scaled.shape, y_train.shape, y_test.shape

logreg = LogisticRegression(solver='lbfgs', max_iter=1000)

logreg.fit(X_train_scaled, y_train)


theta_sklearn = np.concatenate([[logreg.intercept_[0]], logreg.coef_.ravel()])

cost_train_sklearn = compute_cost(X_train_scaled_with_intercept, y_train, theta_sklearn.reshape(-1, 1))
cost_test_sklearn = compute_cost(X_test_scaled_with_intercept, y_test, theta_sklearn.reshape(-1, 1))

cost_train_sklearn, cost_test_sklearn, theta_sklearn[:5]
