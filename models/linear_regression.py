import os
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    def __init__(self, learning_rate=0.1,epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.loss_history = []

    def predict(self, X):
        """Predict value using linear model"""
        return np.dot(X, self.weights) + self.bias
    
    def compute_loss(self, y_true, y_pred):
        """Compute mean squared error"""
        return np.mean(errors**2)
    
    def fit(self, X, y):
        """train the model using Gradient descent"""
        n_samples, n_features = X.shape
        self.weights = np.random.randn(n_features, 1) #initilize weights
        self.bias = 0 #initilize bias

        for epochs in range(self.epochs):
            y_pred = self.predict
            loss = self.compute_loss(y, y_pred)
            self.loss_history.append(loss)

            #Compute gradients
            errors = y_pred - y
            weight_gradient = (2/n_samples) * np.dot(X.T, errors)
            bias_gradients = (2/n_samples) * np.sum(errors)

            #update weights and bias
            self.weights -= self.learning_rate * weight_gradient
            self.bias -= self.learning_rate * bias_gradients
            
            if epochs % 100 == 0:
                print(f"Epoch {epochs}, Loss: {loss:.4f}")
    def plot_loss(self, save_path="loss_curve.png"):
        """plot and save the loss curve"""
        plt.plot(range(self.epochs),self.loss_history,color="green")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Loss over time")
        plt.savefig(save_path,dpi=300,bbox_inches="tight")
        plt.close()