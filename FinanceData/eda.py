import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
import os
import zipfile

"""
# Load the dataset and drop rows
data = pd.read_csv("FinanceData/Dataset/original-dataset.csv").drop(index=np.arange(2429))

# Print dataset information and summary statistics
print("Dataset information:\n", data.info())
print("\nDataset summary statistics:\n", data.describe())

# Check for missing values and visualize missing value pattern
print("\nTotal number of missing values in dataset:", data.isnull().sum().sum())
sns.heatmap(data.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()

# Compute correlation matrix and visualize it
corr = data.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.show()

# Find missing values in the dataset
missing_values = data.isna()

# Impute missing values using cubic spline interpolation
for column in data.columns:
    if missing_values[column].any():
        # Create a cubic spline interpolation function for the current column
        f = interp1d(np.where(~missing_values[column])[0], data[column][~missing_values[column]], kind='cubic', fill_value="extrapolate")
        # Impute missing values in the current column
        data[column][missing_values[column]] = f(np.where(missing_values[column])[0])

# Save the interpolated DataFrame to a CSV file
output_dir = 'FinalDataset2'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'final-dataset2.csv')
data.to_csv(output_path, index=False)

"""

import zipfile

# Directory to compress
directory_to_compress = "/home/noe/GitHub/BTC/FinanceData/Dataset/original-dataset.csv"

# Create a ZipFile object for the output file
output_file = zipfile.ZipFile("original-dataset.zip", "w", zipfile.ZIP_DEFLATED)



# Close the ZipFile object
output_file.close()