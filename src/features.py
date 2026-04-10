import pandas as pd
import numpy as np
import yfinance as yf


def get_market_sentiment():

    spy = yf.download("^GSPC", period="2y")
    spy['SMA_200'] = spy['Close'].rolling(window=200).mean()
    
    current_price = spy['Close'].iloc[-1]
    current_sma = spy['SMA_200'].iloc[-1]
    

    return current_price > current_sma
def add_technical_indicators(series):

    if isinstance(series, pd.DataFrame):
        series = series.squeeze()
        
    df = series.to_frame(name='Close')
    

    df['ret_1d'] = df['Close'].ffill().pct_change()
    df['ret_5d'] = df['Close'].ffill().pct_change(5)
    df['vol_20'] = df['ret_1d'].rolling(window=20).std()
    df['ema_200'] = df['Close'].ewm(span=200).mean()
    df['dist_ema_200'] = (df['Close'] - df['ema_200']) / df['ema_200']

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    return df.dropna()

def create_target(df, horizon=5):

    return df.ffill().pct_change(horizon).shift(-horizon)

def add_global_features(df):
    df = df.copy()
    
    spy_raw = yf.download("^GSPC", period="5y", progress=False)
    spy_ret = spy_raw['Close'].pct_change()
    spy_vol = spy_ret.rolling(20).std()
    df['ret_1d'] = df.groupby('Ticker')['Close'].ffill().transform(lambda x: x.pct_change())
    df['vol_20'] = df.groupby('Ticker')['ret_1d'].transform(lambda x: x.rolling(20).std())
    df['rsi_14'] = df.groupby('Ticker')['Close'].transform(lambda x: calculate_rsi_series(x))
    
    df['ema_200'] = df.groupby('Ticker')['Close'].transform(lambda x: x.ewm(span=200).mean())
    df['dist_ema_200'] = (df['Close'] - df['ema_200']) / df['ema_200']
    
    df['log_ret'] = df.groupby('Ticker')['Close'].transform(lambda x: np.log(x / x.shift(1)))
    df['target'] = df.groupby('Ticker')['log_ret'].transform(lambda x: x.shift(-5).rolling(5).sum())

    df['market_vol'] = spy_vol.reindex(df.index).values
    df['relative_volatility'] = df['vol_20'] / df['market_vol']
    df['Ticker_Cat'] = df['Ticker'].astype('category').cat.codes


    safe_havens = ['GLD', 'TLT', 'SHY']
    df['is_safe_haven'] = df['Ticker'].apply(lambda x: 1 if x in safe_havens else 0)
    
    df['market_ret_5d'] = spy_ret.rolling(5).sum().reindex(df.index).values
    df['alpha_5d'] = df.groupby('Ticker')['ret_1d'].transform(lambda x: x.rolling(5).sum()) - df['market_ret_5d']
    
    return df.dropna()

def calculate_rsi_series(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))