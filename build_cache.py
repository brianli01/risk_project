import pandas as pd
import pickle
from pathlib import Path

from data import assemble_features
from markov import compute_vol, cluster_vol, mkov
from markov_gmm import compute_vol as gmm_compute_vol, cluster_vol_gmm, mkov_soft

# 1) Paths
DATA_CSV         = Path("SPY ETF Stock Price History 93-25.csv")
CACHE_DIR        = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

# 2) Cache daily features
data_cache = CACHE_DIR / "data_features.pkl"
if not data_cache.exists():
    print("Building data features…")
    df, CORE_FEATURES, EXTRA_FEATURES, FEATURES = assemble_features(str(DATA_CSV))
    with open(data_cache, "wb") as f:
        pickle.dump((df, CORE_FEATURES, EXTRA_FEATURES, FEATURES), f)
else:
    print("Data cache already exists.")

# 3) Cache original KMeans regimes
km_cache = CACHE_DIR / "markov_km.pkl"
if not km_cache.exists():
    print("Building KMeans regimes…")
    # load from cache
    with open(data_cache, "rb") as f:
        df, _, _, _ = pickle.load(f)
    df_vol   = compute_vol(df)
    df_km    = cluster_vol(df_vol)
    pmatrix  = mkov(df_km)
    with open(km_cache, "wb") as f:
        pickle.dump((df_km, pmatrix), f)
else:
    print("KMeans cache already exists.")

# 4) Cache GMM regimes
gmm_cache = CACHE_DIR / "markov_gmm.pkl"
if not gmm_cache.exists():
    print("Building GMM regimes…")
    # load raw data again
    with open(data_cache, "rb") as f:
        df, _, _, _ = pickle.load(f)
    df_vol_gmm = gmm_compute_vol(df)
    df_gmm     = cluster_vol_gmm(df_vol_gmm)
    pmatrix_gmm= mkov_soft(df_gmm, [c for c in df_gmm.columns if c.startswith("gamma_")])
    with open(gmm_cache, "wb") as f:
        pickle.dump((df_gmm, pmatrix_gmm), f)
else:
    print("GMM cache already exists.")

print("All caches built.")