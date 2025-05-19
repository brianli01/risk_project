import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# prep data
df = pd.read_csv("SPY ETF Stock Price History 93-25.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df = df.sort_values("Date", ascending=True).copy()   # <-- .copy() here

def compute_vol(df):
    df = df.copy()
    # log returns
    df.loc[:, 'log_ret'] = np.log(df['Price'] / df['Price'].shift(1))

    # realized vols
    windows = [1, 5, 21, 252]
    trading_days = 252
    for w in windows:
        rv     = np.sqrt((df['log_ret']**2).rolling(w).sum())
        rv_ann = np.sqrt(trading_days / w) * rv

        df.loc[:, f'rv_{w}d']     = rv
        df.loc[:, f'rv_{w}d_ann'] = rv_ann

    ann_cols = [f'rv_{w}d_ann' for w in windows]
    df = df.dropna(subset=ann_cols)
    return df

def cluster_vol(df):
    df = df.copy()
    features = ['rv_1d_ann','rv_5d_ann','rv_21d_ann','rv_252d_ann']

    # pull out a clean feature matrix
    X = df[features].dropna().values

    # standardize
    scaler     = StandardScaler()
    X_scaled   = scaler.fit_transform(X)

    # k-means
    km        = KMeans(n_clusters=3, random_state=42)
    labels    = km.fit_predict(X_scaled)
    centroids = km.cluster_centers_

    # map the 3 numeric clusters → low/mod/high by average centroid height
    scores = centroids.mean(axis=1)
    order  = np.argsort(scores)    # [idx_low, idx_mod, idx_high]
    label_map = {
        order[0]: 'low',
        order[1]: 'moderate',
        order[2]: 'high'
    }
    df.loc[df[features].dropna().index, 'vol_cluster'] = [label_map[l] for l in labels]
    df[['Date','vol_cluster']].to_csv('vol_clusters.csv', index=False)
    return df

df_vol       = compute_vol(df)
df_clustered = cluster_vol(df_vol)

vc_df = pd.read_csv("vol_clusters.csv")

def mkov(df):
    df['next_cluster'] = df['vol_cluster'].shift(-1)
    trans = df.dropna(subset=['next_cluster'])

    count_matrix = pd.crosstab(
        trans['vol_cluster'],
        trans['next_cluster']
    )

    prob_matrix = count_matrix.div(count_matrix.sum(axis=1), axis=0)

    return prob_matrix


pmatrix = mkov(vc_df)
print(pmatrix)
