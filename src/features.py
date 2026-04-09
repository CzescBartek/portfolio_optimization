import pandas as pd
import numpy as np

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