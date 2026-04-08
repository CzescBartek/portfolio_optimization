from src.data_loader import download_stock_data
from src.features import add_technical_indicators, create_target
from src.processor import DataProcessor
import os
import pandas as pd

TICKER = "AAPL"
DATA_PATH = "data/processed_aapl.csv"

if __name__ == "__main__":

    raw_data = download_stock_data(["AAPL"])

    
    if isinstance(raw_data.columns, pd.MultiIndex):
        df_close = raw_data['AAPL'].copy()
    else:
        df_close = raw_data.copy()


    df_close = raw_data['AAPL'] if isinstance(raw_data, pd.DataFrame) else raw_data
    df_with_features = add_technical_indicators(df_close)
    df_with_features['target'] = create_target(df_close)
    df_with_features = df_with_features.dropna()

    
    processor = DataProcessor()
    X, y = processor.prepare_data(df_with_features)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)

    print(X_train)