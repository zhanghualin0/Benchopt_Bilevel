from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

import argparse


data = pd.read_parquet('figures/benchopt_ijcnn1.parquet')
data.to_csv('figures/benchopt_ijcnn1.csv', index=False)

# Load the CSV file
csv_file_path = 'figures/benchopt_ijcnn1.csv'  # Replace with your file path
csv_data = pd.read_csv(csv_file_path)

# Extracting only the solver names from the 'solver_name' column
# Assuming the solver names are the first part of the string before the first '[' character
csv_data['simple_solver_name'] = csv_data['solver_name'].str.split('[').str[0]

# Setting up the plot style
sns.set(style="whitegrid")

# Creating the plot
plt.figure(figsize=(8, 6))
sns.lineplot(data=csv_data, x='stop_val', y='objective_value_func', hue='simple_solver_name')

# Adding plot labels and title
plt.xlabel('Iteration')
plt.ylabel('Function Value')
plt.title('Function Value vs Iteration for Different Solvers')
plt.legend(title='Solvers')

# Show the plot
plt.tight_layout()

# Save the figure
output_file_path = 'figures/ijcnn1.pdf' 
plt.savefig(output_file_path)

plt.show()