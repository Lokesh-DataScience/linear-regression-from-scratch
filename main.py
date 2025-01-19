import os
from data_generation.generate_data import generate_data, plot_data
from models.linear_regression import LinearRegression
import matplotlib.pyplot as plt

def main():
    #ensure 'plots' directory exists
    os.makedirs("plots",exist_ok=True)

    X, y = generate_data()
    plot_data(X,y,save_path="plots/data_plot.png")

    #train the linear regression model
    model = LinearRegression(learning_rate=0.1, epochs=2000)
    model.fit(X,y)

    #predict and save fitted line plot
    y_pred = model.predict(X)
    plt.scatter(X, y, color="blue", alpha=0.5,label="Data")
    plt.plot(X,y_pred,color="red",label="Fitted Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.savefig("plots/fitted_line.png",dpi=300,bbox_inches="tight")
    plt.close()

    #save loss and curve plots
    model.plot_loss(save_path="plots/loss_curve.png")

if __name__ == "__main__":
    main()