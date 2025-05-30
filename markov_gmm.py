"""
markov_gmm.py

Refactored functions for computing volatility regimes using Gaussian Mixture Models,
soft clustering, optimal cluster selection, and bootstrapped transition-matrix confidence intervals.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
import pickle
from pathlib import Path

# ——————— Helper functions ———————

def compute_vol(df: pd.DataFrame,
                windows: list = [1, 5, 21, 252],
                trading_days: int = 252) -> pd.DataFrame:
    """
    Compute realized and annualized volatility over specified rolling windows.
    """
    df = df.copy()
    df['log_ret'] = np.log(df['Price'] / df['Price'].shift(1))
    r2 = df['log_ret']**2
    for w in windows:
        s = r2.rolling(w).sum()
        rv = np.sqrt(s)
        rv_ann = np.sqrt(trading_days / w) * rv
        df[f'rv_{w}d']     = rv
        df[f'rv_{w}d_ann'] = rv_ann
    return df.dropna(subset=[f'rv_{w}d_ann' for w in windows])


def select_optimal_gmm(df: pd.DataFrame,
                       windows: list = [1, 5, 21, 252],
                       k_range: range = range(2, 7)) -> int:
    """
    Determine optimal number of clusters via silhouette score on annualized-vol features.
    """
    df_vol = compute_vol(df, windows)
    feats = [f'rv_{w}d_ann' for w in windows]
    X = df_vol[feats].values
    Xs = StandardScaler().fit_transform(X)

    best_k, best_score = None, -np.inf
    for k in k_range:
        gm = GaussianMixture(n_components=k, random_state=42)
        labels = gm.fit_predict(Xs)
        score = silhouette_score(Xs, labels)
        if score > best_score:
            best_k, best_score = k, score
    return best_k


def cluster_vol_gmm(df: pd.DataFrame,
                    n_components: int = 3,
                    windows: list = [1, 5, 21, 252]) -> pd.DataFrame:
    """
    Soft-cluster annualized volatility into regimes using GMM.

    Returns DataFrame with posterior probabilities (gamma) and hard label 'vol_state'.
    """
    df_vol = compute_vol(df, windows)
    feats = [f'rv_{w}d_ann' for w in windows]
    X = df_vol[feats].values
    Xs = StandardScaler().fit_transform(X)

    gm = GaussianMixture(n_components=n_components, random_state=42)
    gm.fit(Xs)
    proba = gm.predict_proba(Xs)
    labels = gm.predict(Xs)

    gammas = pd.DataFrame(
        proba,
        index=df_vol.index,
        columns=[f'gamma_{i}' for i in range(n_components)]
    )
    df_vol = df_vol.join(gammas)
    df_vol['vol_state'] = labels
    return df_vol


def mkov_soft(df: pd.DataFrame, gamma_cols: list) -> pd.DataFrame:
    """
    Compute transition matrix from soft regime probabilities.
    """
    k = len(gamma_cols)
    A = np.zeros((k, k), dtype=float)
    for t in range(len(df) - 1):
        g_t = df.iloc[t][gamma_cols].astype(float).values
        g_next = df.iloc[t+1][gamma_cols].astype(float).values
        A += np.outer(g_t, g_next)
    row_sums = A.sum(axis=1, keepdims=True)
    A = np.divide(A, row_sums, out=np.zeros_like(A), where=row_sums!=0)
    return pd.DataFrame(A, index=gamma_cols, columns=gamma_cols)


def bootstrap_pmatrix(df: pd.DataFrame,
                      gamma_cols: list,
                      n_boot: int = 500,
                      alpha: float = 0.05) -> tuple:
    """
    Bootstrap confidence intervals for each transition probability.
    Returns (lower_df, upper_df).
    """
    from tqdm import trange
    mats = []
    T = len(df)
    df_gamma = df[gamma_cols].astype(float)
    for _ in trange(n_boot, desc="Bootstrapping transition matrix", unit="iter"):
        idx = np.random.randint(0, T-1, size=T-1)
        boot_g = df_gamma.iloc[idx].reset_index(drop=True)
        boot_next = df_gamma.iloc[idx + 1].reset_index(drop=True)
        boot_A = np.zeros((len(gamma_cols), len(gamma_cols)), dtype=float)
        for t in range(len(boot_g)):
            boot_A += np.outer(boot_g.iloc[t].values, boot_next.iloc[t].values)
        row_s = boot_A.sum(axis=1, keepdims=True)
        boot_A = np.divide(boot_A, row_s, out=np.zeros_like(boot_A), where=row_s!=0)
        mats.append(boot_A)
    arr = np.stack(mats)
    lower = np.percentile(arr, 100 * alpha/2, axis=0)
    upper = np.percentile(arr, 100 * (1 - alpha/2), axis=0)
    cols = gamma_cols
    return pd.DataFrame(lower, index=cols, columns=cols), pd.DataFrame(upper, index=cols, columns=cols)

# ——————— Caching logic ———————
CACHE = Path("cache/markov_gmm.pkl")
DATA_CSV = "SPY ETF Stock Price History 93-25.csv"
DATA_CACHE = Path("cache/data_features.pkl")

if CACHE.exists():
    df_gmm, pmatrix_gmm = pickle.load(open(CACHE, "rb"))
else:
    if DATA_CACHE.exists():
        df, *_ = pickle.load(open(DATA_CACHE, "rb"))
    else:
        from data import assemble_features
        df, *_ = assemble_features(DATA_CSV)
    optimal_k     = select_optimal_gmm(df)
    df_gmm        = cluster_vol_gmm(df, n_components=optimal_k)
    gamma_cols    = [c for c in df_gmm.columns if c.startswith("gamma_")]
    pmatrix_gmm   = mkov_soft(df_gmm, gamma_cols)
    CACHE.parent.mkdir(exist_ok=True)
    pickle.dump((df_gmm, pmatrix_gmm), open(CACHE, "wb"))

__all__ = [
    "compute_vol", "select_optimal_gmm", "cluster_vol_gmm",
    "mkov_soft", "bootstrap_pmatrix", "df_gmm", "pmatrix_gmm"
]

# ——————— Script entrypoint ———————
if __name__ == '__main__':
    print("GMM transition matrix:")
    print(pmatrix_gmm)
