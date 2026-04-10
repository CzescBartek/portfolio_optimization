from pypfopt import EfficientFrontier, risk_models
import pandas as pd

class PortfolioOptimizer:
    def __init__(self, prices_df):
        self.prices = prices_df

    def calculate_weights(self, predicted_returns_dict):
        S = risk_models.sample_cov(self.prices)
        mu = pd.Series(predicted_returns_dict)
        
        safe_havens = ['GLD', 'TLT', 'SHY']
        
        try:
            ef = EfficientFrontier(mu, S)
            
            for ticker in mu.index:
                if ticker in safe_havens:
                    ef.add_constraint(lambda w, t=ticker, i=mu.index: w[list(i).index(t)] <= 0.60)
                    ef.add_constraint(lambda w, t=ticker, i=mu.index: w[list(i).index(t)] >= 0.10)
                else:
                    ef.add_constraint(lambda w, t=ticker, i=mu.index: w[list(i).index(t)] <= 0.30)
                    ef.add_constraint(lambda w, t=ticker, i=mu.index: w[list(i).index(t)] >= 0.02)
            
            weights = ef.max_sharpe()
            return ef.clean_weights()
        except Exception:
            ef_min = EfficientFrontier(None, S)
            weights = ef_min.min_volatility()
            return ef_min.clean_weights()