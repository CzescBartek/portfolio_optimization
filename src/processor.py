import pandas as pd
from sklearn.preprocessing import StandardScaler

class DataProcessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None

    def prepare_data(self, df):

        self.feature_cols = [c for c in df.columns if c not in ['target', 'Close']]
        
        X = df[self.feature_cols]
        y = df['target']
        
        X_scaled = self.scaler.fit_transform(X)
        X_scaled_df = pd.DataFrame(X_scaled, columns=self.feature_cols, index=df.index)
        
        return X_scaled_df, y

    def split_data(self, X, y, test_size=0.2):
        split_idx = int(len(X) * (1 - test_size))
        
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test