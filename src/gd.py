import numpy as np

def compute_cost(X, y, theta):
    m = len(y)
    preds = X.dot(theta)
    return (1/(2*m)) * np.sum((preds - y)**2)

def gradient_descent(X, y, theta, lr, iterations):
    m = len(y)
    costs = []

    for _ in range(iterations):
        gradients = (1/m) * X.T.dot(X.dot(theta) - y)
        theta -= lr * gradients
        costs.append(compute_cost(X, y, theta))

    return theta, costs

def mini_batch_gd(X, y, theta, lr, epochs, batch_size=32):
    m = len(y)
    costs = []

    for _ in range(epochs):
        for i in range(0, m, batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]

            gradients = (1/len(y_batch)) * X_batch.T.dot(X_batch.dot(theta) - y_batch)
            theta -= lr * gradients

        costs.append(compute_cost(X, y, theta))

    return theta, costs
