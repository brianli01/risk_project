"""
markov_km.py

Functions for computing volatility regimes using KMeans clustering,
hard clustering into discrete states, and building a Markov transition matrix.
"""
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path

# Helper functions
def compute_vol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute realized and annualized volatility over fixed windows.
    """
    df = df.copy()
    df['log_ret'] = np.log(df['Price'] / df['Price'].shift(1))
    windows = [1, 5, 21, 252]
    trading_days = 252
    for w in windows:
        s = (df['log_ret']**2).rolling(w).sum()
        rv = np.sqrt(s)
        rv_ann = np.sqrt(trading_days / w) * rv
        df[f'rv_{w}d'] = rv
        df[f'rv_{w}d_ann'] = rv_ann
    return df.dropna(subset=[f'rv_{w}d_ann' for w in windows])

def cluster_vol(df: pd.DataFrame) -> pd.DataFrame:
    """
    Hard-cluster annualized vol into low/moderate/high regimes via KMeans.
    """
    df = df.copy()
    feats = ['rv_1d_ann','rv_5d_ann','rv_21d_ann','rv_252d_ann']
    X = df[feats].dropna().values
    Xs = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = km.fit_predict(Xs)
    centroids = km.cluster_centers_
    order = np.argsort(centroids.mean(axis=1))
    label_map = {order[0]:'low', order[1]:'moderate', order[2]:'high'}
    idxs = df[feats].dropna().index
    df.loc[idxs, 'vol_cluster'] = [label_map[l] for l in labels]
    return df

def mkov(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a Markov transition matrix from the hard 'vol_cluster' labels.
    """
    df = df.copy()
    df['next_cluster'] = df['vol_cluster'].shift(-1)
    trans = df.dropna(subset=['vol_cluster','next_cluster'])
    counts = pd.crosstab(trans['vol_cluster'], trans['next_cluster'])
    return counts.div(counts.sum(axis=1), axis=0)

# Caching logic
CACHE = Path("cache/markov_km.pkl")
DATA_CSV = "SPY ETF Stock Price History 93-25.csv"
if CACHE.exists():
    df_km, pmatrix = pickle.load(open(CACHE, "rb"))
else:
    df_raw = pd.read_csv(DATA_CSV, parse_dates=["Date"])
    df_sorted = df_raw.sort_values("Date").reset_index(drop=True)
    df_vol = compute_vol(df_sorted)
    df_km = cluster_vol(df_vol)
    pmatrix = mkov(df_km)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump((df_km, pmatrix), open(CACHE, "wb"))

# Module exports
__all__ = ["compute_vol", "cluster_vol", "mkov", "df_km", "pmatrix"]

# Script entrypoint
if __name__ == "__main__":
    print("KMeans volatility regimes and transition matrix:")
    print(pmatrix)