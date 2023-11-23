import tkinter as tk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import plotly.express as px

class LinearRegressionModel(nn.Module):
    def __init__(self, input_size):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

def load_data():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select CSV File", filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        return df
    else:
        raise ValueError("No file selected.")

def visualize_data(df):
    # Seaborn Pair Plot
    sns.pairplot(df, height=2.5)
    plt.title('Seaborn Pair Plot')
    plt.show()

    # Plotly Scatter Plot
    fig = px.scatter_matrix(df, dimensions=df.columns, title='Plotly Scatter Plot Matrix')
    fig.update_traces(marker=dict(size=3))
    fig.show()

def train_linear_regression(df):
    X = torch.tensor(df['feature'].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(df['target'].values, dtype=torch.float32).view(-1, 1)

    model = LinearRegressionModel(input_size=1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(X)
        loss = criterion(predictions, y)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def visualize_regression(model, df):
    X = torch.tensor(df['feature'].values, dtype=torch.float32).view(-1, 1)
    y = torch.tensor(df['target'].values, dtype=torch.float32).view(-1, 1)

    model.eval()
    with torch.no_grad():
        predictions = model(X)

    plt.scatter(X.numpy(), y.numpy(), label='Actual Data')
    plt.plot(X.numpy(), predictions.numpy(), color='red', label='Linear Regression')
    plt.title('Linear Regression Visualization')
    plt.xlabel('Feature')
    plt.ylabel('Target')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Load Data
    df = load_data()

    # Visualize Data
    visualize_data(df)

    # Linear Regression
    model = train_linear_regression(df[['feature', 'target']])

    # Visualize Regression
    visualize_regression(model, df[['feature', 'target']])
