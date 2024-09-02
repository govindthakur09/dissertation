import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the datasets
df = pd.read_csv('output.csv')
df_subset = pd.read_csv('output_subset.csv')

# Convert numeric columns to numeric types (if needed)
numeric_columns = ["card_number", "Phone 1", "Phone 2", "Email", "IP"]
for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')
    df_subset[column] = pd.to_numeric(df_subset[column], errors='coerce')

# Function to plot distribution plots
def plot_distribution(df, title):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns):
        plt.subplot(3, 2, i+1)
        sns.histplot(df[column].dropna(), kde=True)
        plt.title(f'Distribution of {column}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Function to plot scatter plots
def plot_scatter(df, title):
    sns.pairplot(df)
    plt.suptitle(title, y=1.02)
    plt.show()

# Function to plot box plots
def plot_box(df, title):
    plt.figure(figsize=(15, 10))
    for i, column in enumerate(df.columns):
        plt.subplot(3, 2, i+1)
        sns.boxplot(x=df[column].dropna())
        plt.title(f'Box Plot of {column}')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# Function to plot heatmap
def plot_heatmap(df, title):
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title(title)
    plt.show()

# Plotting for output.csv
plot_distribution(df, 'Distribution Plots for Original Data')
plot_scatter(df, 'Scatter Plots for Original Data')
plot_box(df, 'Box Plots for Original Data')
plot_heatmap(df, 'Heatmap for Original Data')

# Plotting for output_subset.csv
plot_distribution(df_subset, 'Distribution Plots for Subset Data')
plot_scatter(df_subset, 'Scatter Plots for Subset Data')
plot_box(df_subset, 'Box Plots for Subset Data')
plot_heatmap(df_subset, 'Heatmap for Subset Data')
