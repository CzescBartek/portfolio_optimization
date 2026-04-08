import yfinance as yf
import pandas as pd

def download_stock_data(tickers, period="5y"):

    print(f"Pobieranie danych dla: {tickers}...")
    data = yf.download(tickers, period=period)

    return data['Adj Close']