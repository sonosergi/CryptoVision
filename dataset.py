from asset import *



# Define the list of tickers for which we want to download option data
crypto = ['BTC-USD', 'ETH-USD', 'ADA-USD', 'DOGE-USD', 'XRP-USD', 'LTC-USD', 'XLM-USD', 'ETC-USD']    # Cryptocurrencies

stock_tech = ['NVDA', 'AMD', 'INTC', 'TSLA', 'UEN.F']                                                 # Technology stocks

energy_etf = ['REMX', 'SOXX', 'TAN', 'CIBR', 'ARKK', 'LIT', 'BLOK', 'XLE']                            # ETFs related to energy and materials

bonds_etf = ['IEF', 'TLT', 'SHY', 'AGG', 'BND', 'LQD', 'HYG', 'MUB', 'TIP', 'VCIT',                   #
                     'BIV', 'VGIT', 'BLV', 'VGLT', 'VCLT', 'SCHO', 'SCHR', 'SCHZ', 'SPIB',            # Bond ETFs and similar funds
                     'SPAB', 'MBB', 'JNK', 'BKLN', 'PZA', 'BAB', 'ITM', 'MLN', 'HYD', 'EMB']          #

index = ['^GSPC', '^DJI', '^IXIC', '^FTSE', '^GDAXI', '^FCHI', '^N225', '^HSI',                       # 
                      '^BSESN', '^NSEI', '^AXJO', '^GSPTSE', '^BVSP', '^STOXX50E', '^SSMI',           # Stock indices
                      '^IBEX', '^AXKO', '^KS11', '^AXMM', '^AXEJ', '^RUT', '^NYA', '^VIX']            #  

currency = ['EURUSD=X', 'JPY=X', 'GBPUSD=X', 'AUDUSD=X', 'CAD=X', 'CHF=X', 'NZDUSD=X', 
                 'EURJPY=X', 'GBPJPY=X', 'EURGBP=X', 'EURCHF=X', 'HKD=X','SGD=X', 'MXN=X',            #
                 'TRY=X', 'INR=X', 'BRL=X', 'RUB=X','ZAR=X', 'IDR=X', 'THB=X', 'PHP=X', 'KRW=X',      # Currency pairs
                 'USDMYR=X','VND=X', 'TWD=X', 'CLP=X', 'COP=X', 'PEN=X', 'CNY=X']                     #


# Define the start and end dates for the data
start_date = '2011-03-18'
end_date = '2022-12-31'

# Define the number of days to expiry for the options
days_to_expiry = 365

# Loop over each file in the Asset's list
ASSETS = [crypto, stock_tech,  energy_etf, bonds_etf, index, currency]
for file in ASSETS:
    # Loop over each ticker in the list
    for ticker in file:
        # Create an Asset object for the ticker and download its data
        asset = Asset(ticker, start_date, end_date)
        asset.download_data()
        # Compute the stock price volatility and the risk-free rate
        daily_returns = asset.data['Adj Close']
        s = daily_returns.rolling(window=365).std().iloc[-1]
        bond_ticker = '^IRX'
        bond = yf.Ticker(bond_ticker)
        bond_data = bond.history(start=start_date, end=end_date)
        r = bond_data['Open'][-1]
        # Set the pricing model and compute some technical indicators for the Asset object
        asset.black_scholes(days_to_expiry/365, r, s)
        asset.stochastic()
        asset.simple_moving_averages()
        # Save the data for the Asset object to a CSV file
        asset.save_to_csv('Assets')
        # Decrement days_to_expiry and reset to 365 if it reaches 0
        days_to_expiry -= 1
        if days_to_expiry == 0:
            days_to_expiry = 365

# Define the path to the directory where the CSV files are stored
path = 'Assets'

# Create the directory where the output CSV file will be stored if it does not already exist
output_dir = 'Dataset'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load all the CSV files and concatenate them horizontally
files = os.listdir(path)
dfs = []
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, index_col=0)
        dfs.append(df)
result = pd.concat(dfs, axis=1, ignore_index=True)

# Save the cleaned-up DataFrame to a CSV file in the output directory and print it
output_path = os.path.join(output_dir, 'original-dataset.csv')
result.to_csv(output_path, index=False)
print(result)
