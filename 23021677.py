import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the datasets
mens_perfume_path = 'ebay_mens_perfume.csv'
womens_perfume_path = 'ebay_womens_perfume.csv'

mens_data = pd.read_csv(mens_perfume_path)
womens_data = pd.read_csv(womens_perfume_path)

# Combine the datasets
data = pd.concat([mens_data, womens_data], ignore_index=True)

# Data Cleaning
data = data.dropna()
data = data.drop_duplicates()

# Convert prices to include currency symbol
data['price'] = data['price'].apply(lambda x: f"${x:.2f}")

# Select only numerical columns for correlation matrix, excluding 'priceWithCurrency'
numerical_data = data.select_dtypes(include=['float64', 'int64'])

# Summary Statistics
summary_stats = numerical_data.describe()

# Correlation Matrix
correlation_matrix = numerical_data.corr()

# Plotting Functions
def plot_histogram():
    plt.figure(figsize=(10, 6))
    sns.histplot(data['price'].str.replace('$', '').astype(float), kde=True, color='skyblue', bins=30)
    plt.title('Histogram of Prices', fontsize=15)
    plt.xlabel('Price ($)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True)
    plt.show()

def plot_advanced_scatter():
    plt.figure(figsize=(10, 6))
    prices = data['price'].str.replace('$', '').astype(float)
    sold = data['sold']
    scatter = plt.scatter(x=sold, y=prices, c=prices, cmap='viridis', alpha=0.6, edgecolor=None)
    sns.regplot(x=sold, y=prices, scatter=False, color='darkblue')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Price ($)')
    plt.title('Scatter Plot of Items Sold vs. Price', fontsize=15)
    plt.xlabel('Items Sold', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.xscale('log')
    plt.yscale('log')
    plt.grid(True)
    plt.show()

def plot_heatmap():
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='viridis', fmt='.2f')
    plt.title('Heatmap of Correlation Matrix', fontsize=15)
    plt.show()

def plot_box():
    # Get top 10 perfume types by frequency
    top_types = data['type'].value_counts().index[:10]
    filtered_data = data[data['type'].isin(top_types)]
    
    plt.figure(figsize=(12, 8))
    sns.boxplot(x='type', y=filtered_data['price'].str.replace('$', '').astype(float), data=filtered_data, palette='Set2')
    plt.title('Box Plot of Price by Perfume Type', fontsize=15)
    plt.xlabel('Perfume Type', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.grid(True)
    plt.show()

# Call the plotting functions
plot_histogram()
plot_advanced_scatter()
plot_heatmap()
plot_box()

# Displaying the outputs for further use
summary_stats, correlation_matrix
