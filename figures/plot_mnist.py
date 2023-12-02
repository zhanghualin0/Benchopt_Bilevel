from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import argparse


data = pd.read_parquet('figures/benchopt_mnist.parquet')
data.to_csv('figures/benchopt_mnist.csv', index=False)

# Load the CSV file
csv_file_path = 'figures/benchopt_mnist.csv'  # Replace with your file path
csv_data = pd.read_csv(csv_file_path)

# Extracting only the solver names from the 'solver_name' column
# Assuming the solver names are the first part of the string before the first '[' character
csv_data['simple_solver_name'] = csv_data['solver_name'].str.split('[').str[0]

# Setting up the plot style
sns.set(style="whitegrid")

# Creating the plot
plt.figure(figsize=(8, 6))
sns.lineplot(data=csv_data, x='time', y='objective_test_accuracy', hue='simple_solver_name', marker='o')

# Adding plot labels and title
plt.xlabel('Time(sec)')
plt.ylabel('Test error')
plt.xscale('log')
plt.legend(title='Solvers')

# Show the plot
plt.tight_layout()
plt.show()