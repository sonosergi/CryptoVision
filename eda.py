import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.pyplot as plt
import os


# Load the dataset and drop rows
data = pd.read_csv("Dataset/dataset-pre.csv")

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
output_dir = 'Dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
output_path = os.path.join(output_dir, 'final-dataset.csv')
data.to_csv(output_path, index=False)

"""
# Read the original CSV file into a pandas DataFrame
df_original = pd.read_csv('Dataset/dataset-pre.csv', header=None)

# Define the indices of the columns to select
col_indices = [4, 5, 6, 7, 8, 9, 10]

# Initialize an empty DataFrame for the new dataset
df_new = pd.DataFrame()

# Iterate through every 13-column block in the original dataset and select the specified columns
for i in range(0, df_original.shape[1], 13):
    df_block = df_original.iloc[:, i:i+13]
    df_selected = df_block.iloc[:, col_indices]
    df_new = pd.concat([df_new, df_selected], axis=1)

# Save the new dataset to a CSV file
df_new.to_csv('/home/noe/GitHub/BTC/Dataset/final.csv', index=False, header=False)
"""

"""
import zipfile

# Directory to compress
directory_to_compress = "Dataset/original-dataset.csv"

# Create a ZipFile object for the output file
output_file = zipfile.ZipFile("original-dataset.zip", "w", zipfile.ZIP_DEFLATED)



# Close the ZipFile object
output_file.close()

"""

"""
# buscar el índice de la posición donde se encuentra el valor 15929162910 correspondiente al volumen del bitcoin
row_index, col_index = np.where(data == 15929162910)

# imprimir el índice de fila y columna donde se encuentra el valor
print("El valor 15929162910 se encuentra en la posición: ({}, {})".format(row_index[0], col_index[0]))
"""