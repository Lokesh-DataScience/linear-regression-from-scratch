import os
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def predict(self, X):
        """Predict values using the linear model."""
        return np.dot(X, self.weights) + self.bias

    def compute_loss(self, y_true, y_pred):
        """Compute Mean Squared Error."""
        errors = y_true - y_pred
        return np.mean(errors**2)

    def fit(self, X, y):
        """Train the model using Gradient Descent."""
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features, 1)  # Initialize weights
        self.bias = 0  # Initialize bias

        for epoch in range(self.epochs):
            y_pred = self.predict(X)
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            # Compute gradients
            errors = y_pred - y
            weight_gradient = (2 / n_samples) * np.dot(X.T, errors)
            bias_gradient = (2 / n_samples) * np.sum(errors)

            # Update weights and bias
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradient

            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def plot_loss(self, save_path="plots/loss_curve.png"):
        """Plot and save the loss curve."""
        plt.plot(range(self.epochs), self.loss_history, color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss Over Time")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
