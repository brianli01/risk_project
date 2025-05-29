# Revised predict.py with benchmark integrated

import pandas as pd
import numpy as np

import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from markov import pmatrix    # assume pmatrix(series) → (state_probs, trans_mat)
from data import df         

# Drop rows with NaNs introduced by feature engineering
df = df.dropna(subset=[
    'daily_return','rolling_std_5','lag_return_1','volume_5d_avg','rv_next_1'
])

# Define features and target
feature_cols = [
    'daily_return','open_close_range','high_low_range','candle_body_size',
    'upper_wick_size','lower_wick_size','day_of_week','is_month_end',
    'week_number','rolling_std_5','lag_return_1','lag_return_2',
    'volume_5d_avg','volume_change_ratio'
]
X = df[feature_cols].values
y = df['rv_next_1'].values

# Prepare naive benchmark: next-day realized volatility via absolute return
baseline_pred_full = df['daily_return'].abs().values

# Time-series cross-validation
tscv = TimeSeriesSplit(n_splits=5)
for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    # Train model
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        objective='reg:squarederror'
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_model = model.predict(X_test)
    y_pred_naive = baseline_pred_full[test_idx]
    
    # Compute MSEs
    rmse_model = np.sqrt(mean_squared_error(y_test, y_pred_model))
    rmse_naive = np.sqrt(mean_squared_error(y_test, y_pred_naive))
    
    # Compute relative improvement over naive
    improvement = (rmse_naive - rmse_model) / rmse_naive * 100
    
    print(f"Fold {fold} — Model MSE: {rmse_model:.6f}, Naive MSE: {rmse_naive:.6f}, Improvement: {improvement:.2f}%")

