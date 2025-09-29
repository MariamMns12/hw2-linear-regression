import numpy as np
import matplotlib.pyplot as plt

# Mean Squared Error
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

# Train/Validation split
def train_val_split(X, y, train_ratio=0.8, seed=42):
    np.random.seed(seed)
    n = X.shape[0]
    idx = np.random.permutation(n)
    train_size = int(n * train_ratio)
    train_idx, val_idx = idx[:train_size], idx[train_size:]
    return X[train_idx], y[train_idx], X[val_idx], y[val_idx]

# Gradient Descent for Linear Regression
def gradient_descent(Xtr, ytr, Xva, yva, lr=0.01, iters=200):
    m, n = Xtr.shape
    theta = np.zeros(n)

    train_losses = []
    val_losses = []

    for i in range(iters):
        # Predictions
        y_pred = Xtr @ theta
        # Gradient
        grad = (1/m) * (Xtr.T @ (y_pred - ytr))
        # Update
        theta -= lr * grad

        # Track losses
        train_loss = mse(ytr, Xtr @ theta)
        val_loss = mse(yva, Xva @ theta)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

    return theta, train_losses, val_losses

# Plot training vs validation losses
def plot_losses(train_losses, val_losses, title="Loss Curve"):
    plt.figure(figsize=(6,4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title(title)
    plt.legend()
    plt.show()
