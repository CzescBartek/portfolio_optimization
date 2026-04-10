import pandas as pd
import os
import matplotlib.pyplot as plt
from src.data_loader_global import get_global_data # Nowy moduł
from src.features import add_global_features     # Nowy moduł
from src.processor import DataProcessor
from src.model_XG_boost import XGBStockModel
from src.optimizer import PortfolioOptimizer
from src.backtester import Backtester
from src.model_random_forest import StockModel

TICKERS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NFLX','GLD','TLT', 'SHY']
for folder in ['models', 'plots']:
    if not os.path.exists(folder): os.makedirs(folder)

features_to_use = [
    'rsi_14', 
    'vol_20', 
    'relative_volatility',  
    'alpha_5d',             
    'dist_ema_200', 
    'Ticker_Cat'            
]

raw_global_df = get_global_data(TICKERS)
global_df = add_global_features(raw_global_df)

split_date = global_df.index.max() - pd.Timedelta(days=180)
train_df = global_df[global_df.index < split_date]
test_df = global_df[global_df.index >= split_date]

train_df = global_df[global_df.index < split_date]
test_df = global_df[global_df.index >= split_date]

X_train = train_df[features_to_use]
y_train = train_df['target']

X_test = test_df[features_to_use]
y_test = test_df['target']


print("Training global XGBoost...")
model_xgb = XGBStockModel()
model_xgb.train(X_train, y_train)
model_xgb.save('models/global_xgb_model.joblib')

print("Training Global RF...")
model_rf = StockModel()
model_rf.train(X_train, y_train)
model_rf.save('models/global_rf_model.joblib')

latest_data = global_df.groupby('Ticker').tail(1)
X_latest = latest_data[features_to_use]

preds_rf = dict(zip(latest_data['Ticker'], model_rf.predict(X_latest)))
preds_xgb = dict(zip(latest_data['Ticker'], model_xgb.predict(X_latest)))

prices_wide = global_df.pivot(columns='Ticker', values='Close').dropna()
test_prices = prices_wide.loc[prices_wide.index >= split_date]

opt_rf = PortfolioOptimizer(prices_wide)
weights_rf = opt_rf.calculate_weights(preds_rf)

opt_xgb = PortfolioOptimizer(prices_wide)
weights_xgb = opt_xgb.calculate_weights(preds_xgb)

bt_rf = Backtester(test_prices, weights_rf)
cum_rf, benchmark = bt_rf.run()

bt_xgb = Backtester(test_prices, weights_xgb)
cum_xgb, _ = bt_xgb.run()


plt.figure(figsize=(14, 7))
plt.plot(cum_rf, label='Global Random Forest', color='royalblue', alpha=0.8)
plt.plot(cum_xgb, label='Global XGBoost', color='darkorange', linewidth=2)
plt.plot(benchmark, label='Benchmark (Equal Weight)', color='gray', linestyle='--')
plt.title('Global Models Tournament: RF vs XGBoost vs Benchmark')
plt.ylabel('Accumulated Return')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('plots/global_comparison_backtest.png')
plt.show()

