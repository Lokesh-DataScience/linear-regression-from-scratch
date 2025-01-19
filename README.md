# Linear Regression Project (From Scratch)

This project demonstrates the implementation of **Linear Regression** from scratch using Python. It generates synthetic data, trains a linear regression model, and visualizes the results.

## Folder Structure

linear_regression_project/
├── data/
│   └── generate_data.py       # Code to generate synthetic data
├── models/
│   └── linear_regression.py   # Linear regression implementation
├── plots/                     # Directory to save plots
├── main.py                    # Entry point to run the project
├── requirements.txt           # Dependencies
└── README.md                  # Project description

 # Project description
 
## Features

- **Data Generation:** Generates synthetic data with a linear relationship and some noise.
- **Linear Regression Model:** Implements linear regression from scratch using gradient descent.
- **Visualization:** Visualizes and saves the following plots:
  - **Synthetic Data:** A scatter plot of the generated data.
  - **Fitted Line:** The regression line that fits the data.
  - **Loss Curve:** The plot showing how the loss decreases over the training epochs.
  
## Requirements

- Python 3.12.6
- NumPy==2.2.2
- Matplotlib==3.10.0

## Installation

##Clone this repository:

   ```bash
   git clone https://github.com/Lokesh-DataScience/linear-regression-from-scratch.git
   cd linear-regression-from-scratch
   ```
## Create a virtual environment (optional but recommended):
  ```bash
  python -m venv venv
  source venv/bin/activate  # For Linux/MacOS
  venv\Scripts\activate  # For Windows
  ```
## Install the required dependencies:
  ```bash
    pip install -r requirements.txt
  ```
## How to Run ?
**After installation, you can run the project using:**
  ```bash
  python main.py
```
