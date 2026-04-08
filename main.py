from src.data_loader import download_stock_data
from src.features import add_technical_indicators, create_target
from src.processor import DataProcessor
import os
import pandas as pd
from src.model import StockModel


TICKER = "AAPL"
DATA_PATH = "data/processed_aapl.csv"

if __name__ == "__main__":

    raw_data = download_stock_data(["AAPL"])

    
    if isinstance(raw_data.columns, pd.MultiIndex):
        df_close = raw_data['AAPL'].copy()
    else:
        df_close = raw_data.copy()



    df_final = add_technical_indicators(df_close)
    df_final['target'] = create_target(df_close)
    df_final = df_final.dropna()

    
    processor = DataProcessor()
    X, y = processor.prepare_data(df_final)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)


    model = StockModel()
    print("TRAINING MODEL...")
    model.train(X_train, y_train)

    metrics = model.evaluate(X_test, y_test)
    print(f"Model metrics: {metrics}")

    predictions = model.predict(X_test)
    print(f"First 5 forecasts: {predictions[:5]}")
