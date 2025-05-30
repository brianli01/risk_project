# data.py

import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ——————— Helper functions ———————

def load_spy_data(path: str) -> pd.DataFrame:
    """
    Load SPY price data, parse dates, and clean columns.

    Args:
        path: Path to the SPY CSV file.

    Returns:
        DataFrame with raw data and cleaned volume and change columns.
    """
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    # Normalize Volume strings: K→e3, M→e6, B→e9
    df["Volume"] = (
        df["Vol."].str.replace("K", "e3", regex=False)
                  .str.replace("M", "e6", regex=False)
                  .str.replace("B", "e9", regex=False)
                  .astype(float)
    )
    df["Change %"] = df["Change %"].str.replace("%", "", regex=False).astype(float)
    return df


def make_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate core price-based features including log returns,
    open-close range, high-low range, and overnight gap.
    """
    df = df.copy()
    df["daily_return"]     = np.log(df["Price"] / df["Price"].shift(1))
    df["open_close_range"] = df["Open"] - df["Price"]
    df["high_low_range"]   = df["High"] - df["Low"]
    # Overnight gap: returns from prior close to current open
    df["overnight_gap"]    = df["Open"] / df["Price"].shift(1) - 1
    return df


def make_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate core temporal features: day of week, month-end flag, week number.
    """
    df = df.copy()
    df["day_of_week"]  = df["Date"].dt.dayofweek
    df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
    df["week_number"]  = df["Date"].dt.isocalendar().week
    return df


def make_rolling_features(df: pd.DataFrame, windows=None) -> pd.DataFrame:
    """
    Generate rolling and lag features for returns and price.

    Args:
        windows: list of integers for rolling window sizes.
    """
    if windows is None:
        windows = [5, 10, 21, 252]

    df = df.copy()
    for w in windows:
        df[f"roll_ret_std_{w}"]  = df["daily_return"].rolling(w).std()
        df[f"price_ma_{w}"]      = df["Price"].rolling(w).mean()
        df[f"high_roll_max_{w}"] = df["High"].rolling(w).max()
        df[f"low_roll_min_{w}"]  = df["Low"].rolling(w).min()
    # Lagged returns
    df["lag_return_1"] = df["daily_return"].shift(1)
    df["lag_return_2"] = df["daily_return"].shift(2)
    return df


def make_volume_features(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Generate rolling average and change ratio of volume.

    Args:
        window: rolling window size for volume average.
    """
    df = df.copy()
    df["vol_roll_mean"]    = df["Volume"].rolling(window).mean()
    df["vol_change_ratio"] = df["Volume"] / df["Volume"].shift(1)
    return df


def assemble_features(path: str):
    """
    Load raw data, build core and extra feature sets, and define target variables.

    Returns:
        df: DataFrame with all features and targets, NaNs dropped.
        CORE_FEATURES: list of core feature column names.
        EXTRA_FEATURES: list of extra feature column names.
        FEATURES: combined feature list.
    """
    # Step 1: load and clean
    df = load_spy_data(path)

    # Step 2: core feature groups
    df = make_price_features(df)
    df = make_temporal_features(df)
    df = make_rolling_features(df)
    df = make_volume_features(df)

    # Step 3: extra features
    df = df.copy()
    df["candle_body_size"] = (df["Open"] - df["Price"]).abs()
    df["upper_wick_size"]  = df["High"] - df[["Open","Price"]].max(axis=1)
    df["lower_wick_size"]  = df[["Open","Price"]].min(axis=1) - df["Low"]

    # Step 4: target variables
    df["rv_next_1"] = df["daily_return"].abs().shift(-1)
    df["rv_next_5"] = df["daily_return"].rolling(5).std().shift(-5)

    # Step 5: drop NaNs
    df = df.dropna()

    # Feature lists
    CORE_FEATURES = [
        "daily_return", "open_close_range", "high_low_range", "overnight_gap",
        "day_of_week", "is_month_end", "roll_ret_std_5", "vol_roll_mean"
    ]
    EXTRA_FEATURES = [
        "week_number", "price_ma_5", "price_ma_10", "high_roll_max_5", "low_roll_min_5",
        "lag_return_1", "lag_return_2", "candle_body_size", "upper_wick_size",
        "lower_wick_size", "vol_change_ratio"
    ]
    FEATURES = CORE_FEATURES + EXTRA_FEATURES

    return df, CORE_FEATURES, EXTRA_FEATURES, FEATURES


# ——————— Caching logic at import time ———————

CACHE = Path("cache/data_features.pkl")
CSV   = "SPY ETF Stock Price History 93-25.csv"

if CACHE.exists():
    # load precomputed features
    df, CORE_FEATURES, EXTRA_FEATURES, FEATURES = pickle.load(open(CACHE, "rb"))
else:
    # compute and then cache
    df, CORE_FEATURES, EXTRA_FEATURES, FEATURES = assemble_features(CSV)
    CACHE.parent.mkdir(exist_ok=True)
    with open(CACHE, "wb") as f:
        pickle.dump((df, CORE_FEATURES, EXTRA_FEATURES, FEATURES), f)


# ——————— Module exports ———————

__all__ = [
    "load_spy_data", "make_price_features", "make_temporal_features",
    "make_rolling_features", "make_volume_features", "assemble_features",
    "df", "CORE_FEATURES", "EXTRA_FEATURES", "FEATURES"
]


# ——————— Script entrypoint ———————

if __name__ == "__main__":
    print("Data features loaded. Here's a sample:")
    print(df[FEATURES + ["rv_next_1", "rv_next_5"]].head())
