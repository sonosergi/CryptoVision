import yfinance as yf
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
from datetime import datetime
from math import log, sqrt, exp


class Asset:
    
    def __init__(self, ticker, start_date, end_date):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
    
    def descargar_y_guardar(self, ruta_guardado):
        # Descargar los datos
        data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        
        # Crear la ruta deseada (si no existe)
        if not os.path.exists(ruta_guardado):
            os.makedirs(ruta_guardado)

        # Guardar los datos como archivo CSV
        ruta_archivo = os.path.join(ruta_guardado, f"{self.ticker}.csv")
        data.to_csv(ruta_archivo)


class Data(Asset):
    
    def __init__(self, ticker, start_date, end_date, t, r, s):
        super().__init__(ticker, start_date, end_date)
        self.t = t
        self.r = r
        self.s = s

    def BlackScholes(self, CallPutFlag, S, K):
        d1 = (log(S/K) + (self.r + (self.s ** 2)/2) * self.t)/(self.s * sqrt(self.t))
        d2 = d1 - self.s * sqrt(self.t)
        if CallPutFlag == 'c':
            return S * norm.cdf(d1) - K * exp(-self.r * self.t) * norm.cdf(d2)
        else:
            return K * exp(-self.r * self.t) * norm.cdf(-d2) - S * norm.cdf(-d1)
    
    def compute_option_prices(self, df):
        call_prices = []
        put_prices = []

        for index, row in df.iterrows():
            # Extract stock price and option striking price from the dataframe
            S = row['Adj Close']
            K = row['Adj Close']*0.95 # Option striking price is set at 95% of stock price

            # Compute call and put option prices using Black Scholes function
            call_price = self.BlackScholes('c', S, K)
            put_price = self.BlackScholes('p', S, K)

            # Append call and put option prices to the corresponding lists
            call_prices.append(call_price)
            put_prices.append(put_price)

        # Add new columns to the dataframe with call and put option prices
        df['Call Option Price'] = call_prices
        df['Put Option Price'] = put_prices
        
        return df
    
    def save_to_csv(self, df, file_name):
        df.to_csv(file_name, index=False)


# Descargar y guardar los datos de NVDA
nvda = Data("NVDA", "2010-01-01", "2022-01-01", t=1, r=0.02, s=0.3)
nvda.descargar_y_guardar("data")

# Leer los datos de NVDA
nvda_data = pd.read_csv("data/NVDA.csv")

# Seleccionar las columnas de interés
nvda_data = nvda_data[['Open', 'High', 'Low', 'Close', 'Adj Close']]

# Calcular los precios de las opciones y agregarlos al dataframe
nvda_data = nvda.compute_option_prices(nvda_data)

# Guardar los datos en un archivo
nvda.save_to_csv(nvda_data, "nvda_option_prices.csv")

amd = Data("AMD", "2010-01-01", "2022-01-01", t=1, r=0.02, s=0.3)
amd.descargar_y_guardar("data")

# Leer los datos de NVDA
amd_data = pd.read_csv("data/AMD.csv")

# Seleccionar las columnas de interés
amd_data = amd_data[['Open', 'High', 'Low', 'Close', 'Adj Close']]

# Calcular los precios de las opciones y agregarlos al dataframe
amd_data = amd.compute_option_prices(amd_data)

# Guardar los datos en un archivo
amd.save_to_csv(amd_data, "amd_option_prices.csv")