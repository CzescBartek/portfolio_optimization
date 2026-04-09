# Multi-Asset Quantitative Strategy Engine

A machine learning pipeline for stock price prediction and portfolio optimization. This project compares **Local Models** (trained on individual assets) vs. **Global Models** (trained on panel data) using **Random Forest** and **XGBoost** architectures.

## 🚀 Project Overview

The engine automates the entire quantitative workflow:
1.  **Data Ingestion:** Automated fetching of multi-year market data via Yahoo Finance API.
2.  **Feature Engineering:** Generation of technical indicators including RSI, Volatility, and EMA crossovers.
3.  **Modeling:** Comparative analysis between Gradient Boosting (XGBoost) and Bagging (Random Forest) techniques.
4.  **Interpretability:** Utilizing **SHAP (SHapley Additive exPlanations)** to deconstruct model decisions and understand feature importance.
5.  **Optimization:** Mathematical asset allocation using the **Modern Portfolio Theory (Efficient Frontier)** to maximize the Sharpe Ratio.
6.  **Backtesting:** Historical performance evaluation against an Equal-Weight Benchmark.

