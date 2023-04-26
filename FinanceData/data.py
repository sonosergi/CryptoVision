from asset import *


# Define the list of tickers for which we want to download option data
tickers = [
    'BTC-USD', 'NVDA', 'AMD', 'INTC', 'TSLA', 'UBIP',
    "GBTC", "REMX", "SOXX", "TAN", "CIBR", "ARKK", "LIT", "QQQ", "BLOK", "XLE",
    'ETH-USD', 'ADA-USD', 'DOGE-USD', 'XRP-USD', 'BCH-USD', 'DOT1-USD', 'UNI3-USD', 'LINK-USD', 'LTC-USD',
    'SOL1-USD', 'XLM-USD', 'AAVE-USD', 'VET-USD', 'FTT1-USD', 'ETC-USD', 'BNB-USD', 'COMP-USD', 'MATIC-USD',
    'SNX1-USD', 'MKR-USD', 'BAT-USD', 'XTZ-USD', 'KSM1-USD', 'ATOM1-USD', 'GRT2-USD', 'MANA-USD', 'ICP1-USD',
    'DASH-USD', 'FIL1-USD', 'EOS-USD', 'ZEC-USD', 'USDT-USD', 'BTC1=F', 'DX-Y.NYB', 'GLD', 'SLV', 'GDX',
    'HSI1.CME', 'SPX1.CME', 'TY1!', 'CL=F', 'NG=F', 'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'AUDUSD=X', 'CNY=X',
    'XBT', 'GBTC',
    "IEF", "TLT", "SHY", "AGG", "BND", "LQD", "HYG", "MUB", "TIP", "VCIT", "BIV", "VGIT", "BLV", "VGLT", "VCLT", "SCHO", "SCHR", 
    "SCHZ", "SPIB", "SPAB", "MBB", "JNK", "BKLN", "PZA", "BAB", "ITM", "MLN", "HYD", "EMB",
    '^GSPC', '^DJI', '^IXIC', '^FTSE', '^GDAXI', '^FCHI', '^N225', '^HSI', '^SSEC', '^BSESN', '^NSEI', '^AXJO', '^GSPTSE', '^BVSP', 
    '^STOXX50E', '^SSMI', '^FTSEMIB.MI', '^IBEX', '^AXKO', '^KS11', '^AXVI', '^JNIV', '^V2TX', '^AXRE', '^AXMM', '^AXEJ', '^AXYB', 
    '^WORLD', '^MIEF', '^MSCIEF', '^FAW', '^FDEV', '^FEM', '^DWSS', '^RUT', '^NYA', '^VIX', '^VIX9D', '^GDOW', '^XU100.IS'
    'EUR/USD=X', 'USD/JPY=X', 'GBP/USD=X', 'AUD/USD=X', 'USD/CAD=X', 'USD/CHF=X', 'NZD/USD=X', 'EUR/JPY=X', 'GBP/JPY=X', 'EUR/GBP=X', 
    'EUR/CHF=X', 'USD/HKD=X','USD/SGD=X', 'USD/MXN=X', 'USD/TRY=X', 'USD/INR=X', 'USD/BRL=X', 'USD/RUB=X','USD/ZAR=X', 'USD/IDR=X', 
    'USD/THB=X', 'USD/PHP=X', 'USD/KRW=X', 'USD/MYR=X','USD/VND=X', 'USD/TWD=X', 'USD/ARS=X', 'USD/CLP=X', 'USD/COP=X', 'USD/PEN=X',

    ]

# Define the start and end dates for the data
start_date = '2011-03-18'
end_date = '2011-12-31'

# Define the number of days to expiry for the options
days_to_expiry = 365

# Loop over each ticker in the list
for ticker in tickers:
    # Create an Asset object for the ticker and download its data
    asset = Asset(ticker, start_date, end_date)
    asset.download_data()
    
    # Compute the stock price volatility and the risk-free rate
    daily_returns = asset.data['Adj Close']
    s = daily_returns.rolling(window=365).std().iloc[-1]
    bond_ticker = "^IRX"
    bond = yf.Ticker(bond_ticker)
    bond_data = bond.history(start=start_date, end=end_date)
    r = bond_data["Open"][-1]

    # Set the pricing model and compute some technical indicators for the Asset object
    asset.black_scholes(days_to_expiry/365, r, s)
    asset.simple_moving_averages()
    asset.stochastic()
    
    # Save the data for the Asset object to a CSV file
    asset.save_to_csv('data5')

    # Decrement days_to_expiry and reset to 365 if it reaches 0
    days_to_expiry -= 1
    if days_to_expiry == 0:
        days_to_expiry = 365


# Load all the CSV files and concatenate them horizontally
path = 'data5'
files = os.listdir(path)
dfs = []
for file in files:
    if file.endswith('.csv'):
        file_path = os.path.join(path, file)
        df = pd.read_csv(file_path, index_col=0)
        dfs.append(df)
result = pd.concat(dfs, axis=1, ignore_index=True)

# Clean up the concatenated DataFrame by removing the first 756 rows
df_concat = result.drop(result.index[:756], axis=0)

# Save the cleaned-up DataFrame to a CSV file and print it
df_concat.to_csv('dataset.csv', index=False)
print(df_concat)