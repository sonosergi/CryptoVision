import yfinance as yf
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from math import log, sqrt, exp



class Yfinance_download:
    
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



class Option_price:
    
    def __init__(self, t, r, s):
        # Constructor method that initializes the object with the provided arguments
        self.t = t    # t: time to maturity
        self.r = r    # r: risk-free rate
        self.s = s    # s: volatility
        
    def black_scholes(self, call_put_flag, S, K):
        # Method that computes the Black-Scholes formula for European options
        d1 = (log(S/K) + (self.r + (self.s ** 2)/2) * self.t)/(self.s * sqrt(self.t))
        d2 = d1 - self.s * sqrt(self.t)
        # call_put_flag: specifies whether the option is a call or put (input 'c' or 'p')
        if call_put_flag == 'c':
            # Call option price
            return S * norm.cdf(d1) - K * exp(-self.r * self.t) * norm.cdf(d2)    # S: current stock price
        else:
            # Put option price
            return K * exp(-self.r * self.t) * norm.cdf(-d2) - S * norm.cdf(-d1)  # K: strike price



class Asset(Yfinance_download):
    
    def __init__(self, ticker, start_date, end_date):
        super().__init__(ticker, start_date, end_date)
        self.option_pricing_model = None
        self.option_data = None
        
    def set_option_pricing_model(self, option_pricing_model):
        self.option_pricing_model = option_pricing_model
    
    def compute_option_prices(self):
        # Download data if not already downloaded
        if self.data is None:
            self.download_data()
        # Raise error if option pricing model is not set
        if self.option_pricing_model is None:
            raise ValueError("Option pricing model is not set.")
        # Copy the data to a new variable
        data = self.data.copy()
        # Create empty lists to store call and put option prices
        call_prices = []
        put_prices = []
        # Compute call and put option prices for each row in data
        for index, row in data.iterrows():
            # Compute the stock price and strike price
            S = row['Adj Close']
            if 'c':
                K = S * 1.15
            else:
                K = S * 0.85
            # Compute the call and put option prices using the Black-Scholes model
            call_price = self.option_pricing_model.black_scholes('c', S, K)
            put_price = self.option_pricing_model.black_scholes('p', S, K)
            # Append the prices to the respective lists
            call_prices.append(call_price)
            put_prices.append(put_price)
        # Add the call and put option prices to the data frame
        data['Call Option Price'] = call_prices
        data['Put Option Price'] = put_prices
        # Save the option data to the instance variable
        self.option_data = data

    def simple_moving_averages(self):
        # Download data if not already downloaded
        if self.data is None:
            self.download_data()
        # Copy the data to a new variable
        data = self.data.copy()
        # Calculate the 126, 252, and 756-day simple moving averages
        sma_126 = data['Adj Close'].rolling(window=126).mean()
        sma_252 = data['Adj Close'].rolling(window=252).mean()
        sma_756 = data['Adj Close'].rolling(window=756).mean()
        # Combine the moving averages into a single DataFrame
        sma_data = pd.concat([sma_126, sma_252, sma_756], axis=1)
        sma_data.columns = ['SMA 126', 'SMA 252', 'SMA 756']
        return sma_data

    def save_to_csv(self, file_path):
        # Raise error if option data is not computed
        if self.option_data is None:
            raise ValueError("Option data is not computed.")
        # Create the directory if it does not exist
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # Replace '.' in ticker name with '-' and create the file name
        ticker_file_name = self.ticker.replace('.', '-')
        file_name = os.path.join(file_path, f"{ticker_file_name}.csv")
        # Save the option data to a CSV file
        self.option_data.to_csv(file_name, index=True)



if __name__ == '__main__':

    # List of tickers for which we want to download option data
    tickers = ['BTC-USD', 'NVDA', 'AMD', 'INTC']

    # Define the start and end dates for the data
    start_date = '2015-01-01'
    end_date = '2022-12-31'

    days_to_expiry = 252

    # Loop over each ticker in the list
    for ticker in tickers:
        # Create an Asset object for the ticker and download its data
        asset = Asset(ticker, start_date, end_date)
        asset.download_data()
        # Compute the stock price and strike price
        daily_returns = asset.data['Adj Close']
        s = daily_returns.rolling(window=252).std().iloc[-1]


        # Download historical data for the 1-year US Treasury bond (IRX)
        bond_ticker = "^IRX"
        bond = yf.Ticker(bond_ticker)
        bond_data = bond.history(start=start_date, end=end_date)

        # Set r equal to the daily return for the last day in the bond data
        r = bond_data["Open"][-1]

        # Create an OptionData object for the Asset object and set its pricing model
        asset.set_option_pricing_model(Option_price(days_to_expiry/252, r, s))

        # Compute the option prices and save the data to a CSV file in the 'data' directory
        asset.compute_option_prices()
        sma_data = asset.simple_moving_averages()
        asset.option_data = pd.concat([asset.option_data, sma_data], axis=1)
        asset.save_to_csv('data3')

        days_to_expiry -= 1
        if days_to_expiry == 0:
            days_to_expiry = 252