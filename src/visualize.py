import matplotlib.pyplot as plt
import numpy as np
from preprocess import load_and_preprocess
from gd import gradient_descent

X, y, _ = load_and_preprocess("data/insurance.csv")

X = np.c_[np.ones(X.shape[0]), X]
theta = np.zeros(X.shape[1])

learning_rates = [0.001, 0.01, 0.1]
results = {}

for lr in learning_rates:
    _, costs = gradient_descent(X, y, theta.copy(), lr, 200)
    results[lr] = costs

# Plot learning rates
for lr, costs in results.items():
    plt.plot(costs, label=f"lr={lr}")

plt.legend()
plt.title("Learning Rate vs Cost")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.savefig("outputs/plots/lr_vs_cost.png")
plt.clf()

# Actual vs predicted
theta_final, _ = gradient_descent(X, y, theta.copy(), 0.01, 200)
preds = X.dot(theta_final)

plt.scatter(y, preds)
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs Predicted")
plt.savefig("outputs/plots/pred_vs_actual.png")
