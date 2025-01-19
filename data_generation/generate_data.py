import os
import numpy as np
import matplotlib.pyplot as plt

def generate_data():

    np.random.seed(42) #for reproducibility
    X = 2 * np.random.rand(100, 1) #100 samples, single feature
    y = 4 + 3 * X + np.random.randn(100, 1) #linear relation with noise
    return X, y

def plot_data(X, y, save_path="plots/data_plot.png"):
    plt.scatter(X, y, color="blue", alpha=0.5)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Synthetic Data")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    os.makedirs("plots",exist_ok=True)
    X, y = generate_data()
    plot_data(X, y)