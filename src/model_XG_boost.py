from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

class XGBStockModel:
    def __init__(self):
        self.model = XGBRegressor(
            n_estimators=500,
            learning_rate=0.01,
            max_depth=3,
            subsample=0.7,
            colsample_bytree=0.7,
            reg_lambda=10,
            random_state=42
        )

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        return {"MSE": mse, "R2": r2}
    def save(self, path):
        joblib.dump(self.model, path)
        print(f"Model saved in: {path}")

    @staticmethod
    def load(path):
        return joblib.load(path)