import yfinance as yf
import pandas as pd

def get_global_data(tickers, period="5y"):
    data = yf.download(tickers, period=period, group_by='ticker')
    all_dfs = []
    for ticker in tickers:
        temp_df = data[ticker].copy()
        temp_df['Ticker'] = ticker
        all_dfs.append(temp_df)
    return pd.concat(all_dfs)