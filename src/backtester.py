import pandas as pd
import matplotlib.pyplot as plt

class Backtester:
    def __init__(self, prices_df, weights):
        self.prices = prices_df
        self.weights = weights

    def run(self):

        returns = self.prices.pct_change().dropna()
        
        weight_series = pd.Series(self.weights)
        
        portfolio_returns = returns.dot(weight_series)
        
        cumulative_returns = (1 + portfolio_returns).cumprod()
        equal_weights = pd.Series(1/len(self.weights), index=self.weights.keys())
        benchmark_returns = (1 + returns.dot(equal_weights)).cumprod()
        
        return cumulative_returns, benchmark_returns

    def plot_results(self, cumulative, benchmark):
        plt.figure(figsize=(12, 6))
        plt.plot(cumulative, label='ML Strategy (Optimal Weights)', color='green', linewidth=2)
        plt.plot(benchmark, label='Benchmark (Equal Weight)', color='gray', linestyle='--')
        plt.title('Backtest: ML strategy vs Equal distribution')
        plt.xlabel('Data')
        plt.ylabel('Acumulated return')
        plt.legend()
        plt.grid(True)
        plt.show()