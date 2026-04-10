# 📈 Multi-Asset Quantitative Strategy & Portfolio Optimization

A professional-grade Machine Learning pipeline designed to predict stock returns and optimize asset allocation using Global Panel Data, Gradient Boosting (XGBoost), and Explainable AI (XAI).

## 🌟 Key Features
* **Global Model Architecture:** Trains on 100,000+ data points across multiple sectors to learn universal market dynamics instead of ticker-specific noise.
* **Multi-Asset Strategy:** Expanded universe beyond Equities to include **Safe Havens** (Gold, 20yr+ Treasuries, Short-term Bonds) for downside protection.
* **Advanced Feature Engineering:**
    * `alpha_5d`: Cross-sectional momentum (Individual Return vs. S&P 500 Return).
    * `relative_volatility`: Asset risk normalized against overall Market Volatility.
    * `dist_ema_200`: Percentage distance from the 200-day trend line.
* **Explainable AI (XAI):** Integrated **SHAP** analysis to visualize the "why" behind every buy/sell signal and asset rotation.
* **Risk-Aware Optimization:** Enhanced **Efficient Frontier** construction that dynamically shifts capital to defensive assets when global equity forecasts are bearish.

## 🏗️ System Architecture
The project is built with a modular structure for production-readiness:
* `src/`: Core engine containing Data Loaders, Feature Engineering, and Model Wrappers.
* `models/`: Persistent storage for trained Global XGBoost and Random Forest models (.joblib).
* `plots/`: Automated visualization export for SHAP analysis and Backtest results.
* `main_global.py`: The primary pipeline for large-scale panel data training.
* `live_predictor.py`: Real-time inference engine generating daily signals and currency-based position sizing.

## 📊 Model Performance & Comparison

The system evaluates two distinct architectures. By introducing `alpha_5d` and `relative_volatility`, the XGBoost model achieved significant decoupling from market downturns.

### Battle of the Models: Global RF vs Global XGBoost (with Alpha Features)

| Metric | Random Forest (Global) | XGBoost (Global + Alpha) | Winner |
| :--- | :--- | :--- | :--- |
| **Backtest Return** | -12.4% | **+22.8%** | **XGBoost** |
| **Alpha vs Benchmark** | Negative | **+26.5%** | **XGBoost** |
| **Market Correlation** | High (Beta ~1.1) | **Low (Beta ~0.4)** | **XGBoost** |
| **Safe Haven Allocation**| Static | **Dynamic (Up to 80%)** | **XGBoost** |

**Quant Insight:** The XGBoost model successfully captured cross-sectional momentum. During the Q1 2026 market correction, the model generated substantial Alpha by automatically rotating the portfolio into **GLD** and **TLT**, effectively hedging against tech sector volatility.

### Strategy Visualization

<p align="center">
  <img src="plots/global_comparison_backtest.png" width="90%" alt="Backtest Comparison" />
</p>
<p align="center">
  <em>Historical Backtest: Global XGBoost (Gold) successfully outperforming the Benchmark (Gray) during high volatility regimes.</em>
</p>

---

