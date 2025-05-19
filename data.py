import pandas as pd
import numpy as np

# Load SPY data
# Replace with your local CSV path if needed
df = pd.read_csv(
    "SPY ETF Stock Price History 93-25.csv",
    parse_dates=["Date"]
)
df = df.sort_values("Date").reset_index(drop=True)

# -- Data Cleaning --
# Normalize Volume strings: K->e3, M->e6, B->e9
df["Volume"] = (
    df["Vol."].
    str.replace("K", "e3", regex=False).
    str.replace("M", "e6", regex=False).
    str.replace("B", "e9", regex=False).
    astype(float)
)
df["Change %"] = df["Change %"].str.replace("%", "", regex=False).astype(float)

# -- Core Features --
# 1. Price-based core features
df["daily_return"]      = np.log(df["Price"] / df["Price"].shift(1))
df["open_close_range"]  = df["Open"] - df["Price"]
df["high_low_range"]    = df["High"] - df["Low"]

# 2. Temporal core features
df["day_of_week"]       = df["Date"].dt.dayofweek
df["is_month_end"]      = df["Date"].dt.is_month_end.astype(int)

# 3. Rolling/lag core features
df["rolling_std_5"]     = df["daily_return"].rolling(5).std()

# 4. Volume core features
df["volume_5d_avg"]     = df["Volume"].rolling(5).mean()

# -- Extra Features --
# 1. Extended price-based features
df["candle_body_size"]  = (df["Open"] - df["Price"]).abs()
df["upper_wick_size"]   = df["High"] - df[["Open","Price"]].max(axis=1)
df["lower_wick_size"]   = df[["Open","Price"]].min(axis=1) - df["Low"]

# 2. Extended temporal features
df["week_number"]       = df["Date"].dt.isocalendar().week

# 3. Additional rolling features
df["rolling_std_10"]    = df["daily_return"].rolling(10).std()
df["rolling_mean_5"]    = df["Price"].rolling(5).mean()
df["rolling_high_5"]    = df["High"].rolling(5).max()
df["rolling_low_5"]     = df["Low"].rolling(5).min()
df["lag_return_1"]      = df["daily_return"].shift(1)
df["lag_return_2"]      = df["daily_return"].shift(2)
df["rolling_volatility_spread"] = df["rolling_std_5"] / df["rolling_mean_5"]

# 4. Extended volume features
df["volume_change_ratio"] = df["Volume"] / df["Volume"].shift(1)

# -- Target Variable --
df["rv_next_5"]         = df["daily_return"].rolling(5).std().shift(-5)

df['rv_next_1'] = df['daily_return'].abs().shift(-1)
# Drop rows with NaNs from feature engineering
df = df.dropna()

# -- Feature Lists --
CORE_FEATURES = [
    "daily_return", "open_close_range", "high_low_range",
    "day_of_week", "is_month_end", "rolling_std_5", "volume_5d_avg"
]

EXTRA_FEATURES = [
    "candle_body_size", "upper_wick_size", "lower_wick_size", "week_number",
    "rolling_std_10", "rolling_mean_5", "rolling_high_5", "rolling_low_5",
    "lag_return_1", "lag_return_2", "rolling_volatility_spread",
    "volume_change_ratio"
]

# Combined feature set for modeling
FEATURES = CORE_FEATURES + EXTRA_FEATURES

# Optionally export features to CSV for quick inspection
# df[FEATURES + ["rv_next_5"]].to_csv("spy_features.csv", index=False)