
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
import pandas as pd

class PortfolioOptimizer:
    def __init__(self, prices_df):
        self.prices = prices_df

    def calculate_weights(self, predicted_returns_dict):
        S = risk_models.sample_cov(self.prices)
        mu = pd.Series(predicted_returns_dict)
        
        try:
            ef = EfficientFrontier(mu, S)
            ef.add_constraint(lambda w: w <= 0.40)
            ef.add_constraint(lambda w: w >= 0.05)
            weights = ef.max_sharpe()
            return ef.clean_weights()
        except Exception as e:

            ef_min = EfficientFrontier(None, S) 
            ef_min.add_constraint(lambda w: w <= 0.40)
            ef_min.add_constraint(lambda w: w >= 0.05)
            weights = ef_min.min_volatility()
            return ef_min.clean_weights()