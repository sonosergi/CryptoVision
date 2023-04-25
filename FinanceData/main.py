from asset import *
from asset import fibonacci_levels_and_trend

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
    asset.fibonacci_levels_and_trend()
    asset.save_to_csv('data4')

    days_to_expiry -= 1
    if days_to_expiry == 0:
        days_to_expiry = 252