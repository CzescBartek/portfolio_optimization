import joblib
import pandas as pd
from src.data_loader import download_stock_data
from src.features import add_technical_indicators

def get_live_signals(tickers):
    signals = {}
    prices = {}
    
    for ticker in tickers:

        data = download_stock_data([ticker], period="2y")
        prices[ticker] = data[ticker]
        
        df_features = add_technical_indicators(data[ticker])
        latest_row = df_features.tail(1) 
        
        model = joblib.load(f'models/xgb_{ticker}.joblib')
        pred = model.predict(latest_row)[0]
        signals[ticker] = pred
        
    return signals, pd.DataFrame(prices).dropna()

