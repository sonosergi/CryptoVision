import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os


df = pd.read_csv("/home/noe/GitHub/BTC/FinanceData/Dataset/original-dataset.csv")
#print(df.info())
#print(df.describe())
#print(df.isnull().sum().sum())
sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap='viridis')
plt.show()


corr = df.corr()
plt.figure(figsize=(20,20))
sns.heatmap(corr, annot=False, cmap='coolwarm')
plt.show()
