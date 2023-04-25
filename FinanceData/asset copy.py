import yfinance as yf
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from math import log, sqrt, exp



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
  
    def simple_moving_averages(self):
        sma_126 = self.data['Adj Close'].rolling(window=126).mean()
        sma_252 = self.data['Adj Close'].rolling(window=252).mean()
        sma_756 = self.data['Adj Close'].rolling(window=756).mean()
        self.data['SMA 126'] = sma_126
        self.data['SMA 252'] = sma_252
        self.data['SMA 756'] = sma_756

    def calculate_trend(self):
            periods = [256, 504, 756]
            for i, row in self.data.iterrows():
                trends = []
                for period in periods:
                    start_date = i - pd.Timedelta(days=period)
                    end_date = i
                    trend_data = self.data.loc[start_date:end_date]
                    if trend_data.empty:
                        continue
                    trend_high = trend_data['Adj Close'].idxmax()
                    trend_low = trend_data['Adj Close'].idxmin()
                    if trend_high < trend_low:
                        trend = 'Down'
                    else:
                        trend = 'Up'
                    trends.append(trend)
                if 'Down' in trends:
                    self.data.loc[i, 'Trend'] = 'Down'
                else:
                    self.data.loc[i, 'Trend'] = 'Up'
            return self.data

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
        
        asset.black_scholes(days_to_expiry/252, r, s)
        asset.simple_moving_averages()
        asset.calculate_trend()
        asset.save_to_csv('data4')

        days_to_expiry -= 1
        if days_to_expiry == 0:
            days_to_expiry = 252