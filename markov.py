import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def load_spy_data(path: str) -> pd.DataFrame:
    """
    Load SPY price data, parse dates, and return sorted DataFrame.

    Args:
        path: Path to the SPY CSV file.

    Returns:
        DataFrame with Date parsed and sorted.
    """
    df = pd.read_csv(path, parse_dates=["Date"] )
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def compute_realized_volatility(
    df: pd.DataFrame,
    windows: list = [1, 5, 21, 252],
    trading_days: int = 252
) -> pd.DataFrame:
    """
    Calculate realized volatility over multiple windows and annualize.

    Args:
        df: DataFrame with at least 'Price' column.
        windows: List of lookback window sizes in days.
        trading_days: Number of trading days per year for annualization.

    Returns:
        DataFrame with added columns 'rv_{w}d' and 'rv_{w}d_ann' for each window.
    """
    df = df.copy()
    df['log_ret'] = np.log(df['Price'] / df['Price'].shift(1))

    for w in windows:
        # raw realized vol: sqrt(sum of squared returns)
        rv = np.sqrt((df['log_ret']**2).rolling(w).sum())
        # annualized vol
        rv_ann = rv * np.sqrt(trading_days / w)
        df[f'rv_{w}d'] = rv
        df[f'rv_{w}d_ann'] = rv_ann

    # drop initial NaNs
    ann_cols = [f'rv_{w}d_ann' for w in windows]
    df = df.dropna(subset=ann_cols)
    return df


def cluster_volatility(
    df: pd.DataFrame,
    ann_vol_cols: list,
    n_clusters: int = 5,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Cluster days into volatility regimes using KMeans on annualized vol features.

    Args:
        df: DataFrame containing annualized volatility columns.
        ann_vol_cols: List of column names to use for clustering.
        n_clusters: Number of clusters for KMeans.
        random_state: Random seed for reproducibility.

    Returns:
        DataFrame with new 'vol_cluster' column of regime labels.
    """
    df = df.copy()
    # extract and scale features
    X = df[ann_vol_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # fit KMeans
    km = KMeans(n_clusters=n_clusters, random_state=random_state)
    labels = km.fit_predict(X_scaled)
    centers = km.cluster_centers_.mean(axis=1)

    # map clusters to ordered regime names
    order = np.argsort(centers)
    label_names = ['low', 'low-mod', 'moderate', 'mod-high', 'high'][:n_clusters]
    label_map = {order[i]: label_names[i] for i in range(n_clusters)}

    # assign labels
    df['vol_cluster'] = [label_map[l] for l in labels]
    return df


def compute_transition_matrix(
    df: pd.DataFrame,
    cluster_col: str = 'vol_cluster'
) -> pd.DataFrame:
    """
    Compute the Markov transition matrix from cluster labels.

    Args:
        df: DataFrame with a cluster column and sorted by date.
        cluster_col: Name of the column with cluster labels.

    Returns:
        DataFrame representing the transition probability matrix.
    """
    df = df.copy()
    # next-state label
    df['next_cluster'] = df[cluster_col].shift(-1)
    # drop last row
    valid = df.dropna(subset=['next_cluster'])
    counts = pd.crosstab(valid[cluster_col], valid['next_cluster'])
    probs = counts.div(counts.sum(axis=1), axis=0)
    return probs


def assemble_markov_matrix(
    path: str,
    windows: list = [1,5,21,252],
    n_clusters: int = 5
) -> pd.DataFrame:
    """
    Full pipeline: load data, compute vol, cluster regimes, and return transition matrix.

    Args:
        path: CSV file path for SPY prices.
        windows: Lookback windows for realized volatility.
        n_clusters: Number of volatility regimes.

    Returns:
        Transition matrix DataFrame.
    """
    df = load_spy_data(path)
    df_vol = compute_realized_volatility(df, windows)
    ann_cols = [f'rv_{w}d_ann' for w in windows]
    df_clustered = cluster_volatility(df_vol, ann_cols, n_clusters)
    trans_mat = compute_transition_matrix(df_clustered)
    return trans_mat


if __name__ == '__main__':
    matrix = assemble_markov_matrix('SPY ETF Stock Price History 93-25.csv')
    print('Transition matrix:')
    print(matrix)