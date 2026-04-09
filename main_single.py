import pandas as pd
import matplotlib.pyplot as plt
from src.shap import explain_model
from src.data_loader import download_stock_data
from src.features import add_technical_indicators, create_target
from src.processor import DataProcessor
from src.model_random_forest import StockModel
from src.model_XG_boost import XGBStockModel
from src.optimizer import PortfolioOptimizer
from src.backtester import Backtester


TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
preds_rf = {}
preds_xgb = {}
all_prices = {}
metrics_table = []

for ticker in TICKERS:
    raw_data = download_stock_data([ticker])
    df_close = raw_data[ticker]
    all_prices[ticker] = df_close
    
    df_final = add_technical_indicators(df_close)
    df_final['target'] = create_target(df_close)
    df_final = df_final.dropna()
    
    processor = DataProcessor()
    X, y = processor.prepare_data(df_final)
    X_train, X_test, y_train, y_test = processor.split_data(X, y)
    
    model_rf = StockModel()
    model_rf.train(X_train, y_train)
    m_rf = model_rf.evaluate(X_test, y_test)
    preds_rf[ticker] = model_rf.predict(X.tail(1))[0]
    
    model_xgb = XGBStockModel()
    model_xgb.train(X_train, y_train)
    m_xgb = model_xgb.evaluate(X_test, y_test)
    preds_xgb[ticker] = model_xgb.predict(X.tail(1))[0]
    model_rf.save(f'models/rf_{ticker}.joblib')
    model_xgb.save(f'models/xgb_{ticker}.joblib')
    metrics_table.append({'Ticker': ticker, 'RF_R2': m_rf['R2'], 'XGB_R2': m_xgb['R2']})
    explain_model(model_xgb, X_test, 'XGB', ticker)
    explain_model(model_rf, X_test, 'RF', ticker)

print(pd.DataFrame(metrics_table))



prices_df = pd.DataFrame(all_prices).dropna()
test_prices = prices_df.loc[X_test.index]

opt_rf = PortfolioOptimizer(prices_df)
weights_rf = opt_rf.calculate_weights(preds_rf)

opt_xgb = PortfolioOptimizer(prices_df)
weights_xgb = opt_xgb.calculate_weights(preds_xgb)

bt_rf = Backtester(test_prices, weights_rf)
cum_rf, benchmark = bt_rf.run()

bt_xgb = Backtester(test_prices, weights_xgb)
cum_xgb, _ = bt_xgb.run()

plt.figure(figsize=(14, 7))
plt.plot(cum_rf, label='Random Forest', color='blue', alpha=0.7)
plt.plot(cum_xgb, label='XGBoost', color='green', linewidth=2)
plt.plot(benchmark, label='Benchmark (Equal Weight)', color='gray', linestyle='--')
plt.title('Comparison: RF vs XGBoost vs Benchmark')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/backtest_comparison.png')
plt.show()
plt.close()


