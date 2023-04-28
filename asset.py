import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from math import log, sqrt, exp
import os


# Builder of an asset
class Asset():
    # Initializing the class instance with Ticker, Start date and End date
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.data = None

    # Method to download the asset data from Yahoo Finance
    def download_data(self):
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        # Set index to datetime
        self.data.index = pd.to_datetime(self.data.index)
        # Create a range of dates from start_date to end_date
        date_range = pd.date_range(start=self.start_date, end=self.end_date)
        # Reindex with the complete date range
        self.data = self.data.reindex(date_range)
        # Forward fill missing values
        self.data = self.data.fillna(method='ffill')
        return self.data
        
    # Method for calculating the price of put and call options with the Black-scholes equation    
    def black_scholes(self, t, r, s):
        # Create empty lists to store call and put option prices
        call_prices = []
        put_prices = []
        # Compute call and put option prices for each row in data
        for index, row in self.data.iterrows():
            # Compute the stock price and strike price
            S = row['Adj Close'] 
            if 'c':
                K = S * 1.15
            else:
                K = S * 0.85
            # Compute the call and put option prices using the Black-Scholes model
            d1 = (log(S/K) + (r + (s ** 2)/2) * t)/(s * sqrt(t))
            d2 = d1 - s * sqrt(t)
            # call_put_flag: specifies whether the option is a call or put (input 'c' or 'p')
            call_price = S * norm.cdf(d1) - K * exp(-r * t) * norm.cdf(d2)
            put_price = K * exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1)
            # Append the prices to the respective lists
            call_prices.append(call_price)
            put_prices.append(put_price)
        # Add the call and put option prices to the data frame
        self.data['Call Price'] = call_prices
        self.data['Put Price'] = put_prices

    # Method to calculate the moving averages of 183, 365 and 730 days  
    def simple_moving_averages(self):
        sma_183 = self.data['Adj Close'].rolling(window=183).mean()
        sma_365 = self.data['Adj Close'].rolling(window=365).mean()
        sma_1095 = self.data['Adj Close'].rolling(window=730).mean()
        self.data['SMA 183'] = sma_183
        self.data['SMA 365'] = sma_365
        self.data['SMA 730'] = 730

    # Method to calculate Stochastic
    def stochastic(self, k_period=14, d_period=3):
        # Create 'L'owest and 'H'ighest columns
        self.data['L'] = self.data['Low'].rolling(k_period).min()
        self.data['H'] = self.data['High'].rolling(k_period).max()
        # Calculate the %K
        self.data['%K'] = ((self.data['Close'] - self.data['L']) / (self.data['H'] - self.data['L'])) * 100
        # Calculate the %D
        self.data['%D'] = self.data['%K'].rolling(d_period).mean()
        # Drop the 'L' and 'H' columns
        self.data.drop(['L', 'H'], axis=1, inplace=True)
        return self.data

    # Method to save csv file
    def save_to_csv(self, file_path):
        # Raise error if option data is not computed
        if self.data is None:
            raise ValueError("Option data is not computed.")
        # Create the directory if it does not exist
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # Replace '.' in ticker name with '-' and create the file name
        ticker_file_name = self.ticker.replace('.', '-')
        file_name = os.path.join(file_path, f"{ticker_file_name}.csv")
        # Save the option data to a CSV file
        self.data.to_csv(file_name, index=True)