import numpy as np
import joblib
from preprocess import load_and_preprocess
from gd import gradient_descent
from sklearn.linear_model import Ridge, Lasso

# Load data
X, y, feature_names = load_and_preprocess("data/insurance.csv")

# Add bias
X = np.c_[np.ones(X.shape[0]), X]
theta = np.zeros(X.shape[1])

# Train with multiple learning rates
learning_rates = [0.001, 0.01, 0.1]
results = {}

for lr in learning_rates:
    theta_final, costs = gradient_descent(X, y, theta.copy(), lr, 200)
    results[lr] = costs

# Save best model (using sklearn for deployment)
ridge = Ridge(alpha=1.0)
ridge.fit(X[:, 1:], y)

joblib.dump(ridge, "outputs/model.joblib", protocol=2)

print("Training complete. Model saved.")
