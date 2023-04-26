import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

data = pd.read_csv('~/GitHub/BTC/FinanceData/dataset.csv')

X = data.iloc[:, :-1]
y = data.iloc[:, -1]
