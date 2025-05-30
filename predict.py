"""
predict_merged.py

Performs a full machine learning pipeline including data preparation,
ablation studies for feature selection, model training, and diagnostic evaluations.
"""
import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error # MAE kept from original script 1
from scipy import stats

# 0) Setup Cache Path
CACHE = Path("cache")

# 1) Load cached data & regimes
print("1) Loading cached data & regimes...")
# daily features + target
with open(CACHE / "data_features.pkl", "rb") as f:
    df, CORE_FEATURES, EXTRA_FEATURES, FEATURES = pickle.load(f)

# KMeans regimes
with open(CACHE / "markov_km.pkl", "rb") as f:
    df_km, pmatrix_km = pickle.load(f)

# GMM regimes
with open(CACHE / "markov_gmm.pkl", "rb") as f:
    df_gmm, pmatrix_gmm = pickle.load(f)

# 2) Merge regimes into main DataFrame
print("2) Merging regimes...")
df_all = df.copy()
df_all = df_all.merge(df_km[["Date", "vol_cluster"]], on="Date", how="left")
df_all = df_all.merge(df_gmm[["Date", "vol_state"]], on="Date", how="left")


# 3) Build KMeans & GMM transition‐prob features
print("3) Building transition-probability features...")
KM_FEATS = []
for state in pmatrix_km.columns:
    col = f"trans_km_to_{state}"
    df_all[col] = df_all["vol_cluster"].map(pmatrix_km[state])
    KM_FEATS.append(col)

num_gmm_states = len(pmatrix_gmm.columns)
int_to_gamma_map = {i: f'gamma_{i}' for i in range(num_gmm_states)}
df_all['vol_state_str_label'] = df_all['vol_state'].map(int_to_gamma_map)

GMM_FEATS = []
# pmatrix_gmm.columns are the target states, e.g., 'gamma_0', 'gamma_1'
# The index of pmatrix_gmm[target_state_name] are the 'from' states, also 'gamma_0', 'gamma_1'
for target_state_name in pmatrix_gmm.columns:
    col = f"trans_gmm_to_{target_state_name}"
    # Now use 'vol_state_str_label' for mapping
    df_all[col] = df_all["vol_state_str_label"].map(pmatrix_gmm[target_state_name])
    GMM_FEATS.append(col)

# 4) Final feature list & clean data
print("4) Preparing and cleaning data for consistent splitting...")
ALL_MODEL_FEATURES = FEATURES + KM_FEATS + GMM_FEATS # All features that could be used by any model
# Columns needed for target, diagnostics, regime definitions, and all model features
ESSENTIAL_COLUMNS = ["rv_next_1", "daily_return", "vol_cluster", "vol_state"]
df_clean = df_all.dropna(subset=ESSENTIAL_COLUMNS + ALL_MODEL_FEATURES)


# 5) Build arrays & time‐ordered split
print("5) Building arrays & time-ordered split...")
y_full = df_clean["rv_next_1"].values

n = len(df_clean)
split_idx = int(0.8 * n)

y_dev, y_test = y_full[:split_idx], y_full[split_idx:]

# DataFrames for dev and test sets, containing all necessary columns
df_dev_clean = df_clean.iloc[:split_idx]
df_test_clean = df_clean.iloc[split_idx:]

# 6) Hyperparameter tuning
print("6) Hyperparameter tuning on baseline features...")
X_baseline_dev = df_dev_clean[FEATURES].values

tscv = TimeSeriesSplit(n_splits=5)
param_grid = {
    "n_estimators":  [100, 200],
    "max_depth":     [3, 4],
    "learning_rate": [0.01, 0.05],
}
base_xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
search = GridSearchCV(
    estimator=base_xgb_model,
    param_grid=param_grid,
    cv=tscv,
    scoring="neg_mean_squared_error",
    n_jobs=-1,
    verbose=1,
)
search.fit(X_baseline_dev, y_dev)
best_params = search.best_params_
best_dev_rmse_baseline = np.sqrt(-search.best_score_)
print(f"   Best Dev RMSE (baseline features): {best_dev_rmse_baseline:.6f} with params {best_params}")

# 7) Ablation study
print("7) Ablation Study — evaluating feature sets:")
scenarios = {
    "Baseline":        FEATURES,
    "KMeans Markov":   FEATURES + KM_FEATS,
    "GMM Markov":      FEATURES + GMM_FEATS,
    "Combined Markov": FEATURES + KM_FEATS + GMM_FEATS,
}

ablation_results = {}
trained_models = {}
model_predictions = {}
model_feature_lists = {}

for name, feat_list in scenarios.items():
    X_dev_scenario = df_dev_clean[feat_list].values
    X_test_scenario = df_test_clean[feat_list].values
    
    model = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        **best_params
    )
    model.fit(X_dev_scenario, y_dev)
    y_pred = model.predict(X_test_scenario)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    ablation_results[name] = rmse
    trained_models[name] = model
    model_predictions[name] = y_pred
    model_feature_lists[name] = feat_list
    print(f"   Scenario '{name}' Test RMSE: {rmse:.6f}")

# 8) Select best model
best_scenario_name = min(ablation_results, key=ablation_results.get)
best_model_overall = trained_models[best_scenario_name]
y_pred_best_global = model_predictions[best_scenario_name] 
rmse_best_global = ablation_results[best_scenario_name]    
best_feature_list = model_feature_lists[best_scenario_name] 

print(f"\n8) Best model from ablation: '{best_scenario_name}' with Test RMSE: {rmse_best_global:.6f}")

# --- Start of Diagnostics Section ---

# 9) Train per‐KMeans-regime models
print("\n9) Training per-KMeans-regime models using best feature set...")
models_reg_km = {}
reg_dev_km = df_dev_clean["vol_cluster"].values 
X_dev_best_features = df_dev_clean[best_feature_list].values

for state in np.unique(reg_dev_km): # reg_dev_km should be clean of NaNs from step 4
    mask = (reg_dev_km == state)
    if np.sum(mask) > 0: 
        m = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            **best_params
        )
        m.fit(X_dev_best_features[mask], y_dev[mask])
        models_reg_km[state] = m
    else: # Should ideally not be hit if reg_dev_km is not empty and np.unique works
        print(f"   Skipping KMeans regime '{state}' for per-regime model training: no data in dev set.")

reg_test_km = df_test_clean["vol_cluster"].values 
X_test_best_features = df_test_clean[best_feature_list].values
y_pred_reg_km = np.zeros_like(y_test, dtype=float)
valid_predictions_count_km = 0
per_km_regime_models_trained = bool(models_reg_km)

if per_km_regime_models_trained:
    for i, r_test_km in enumerate(reg_test_km): # r_test_km should be clean of NaNs
        if r_test_km in models_reg_km:
            y_pred_reg_km[i] = models_reg_km[r_test_km].predict(X_test_best_features[i].reshape(1, -1))[0]
            valid_predictions_count_km +=1
        else:
            print(f"   Warning: KMeans Regime '{r_test_km}' in test set not found in trained per-KMeans-regime models. Using global prediction for instance {i}.")
            y_pred_reg_km[i] = y_pred_best_global[i] 

    if valid_predictions_count_km > 0:
        rmse_reg_km = np.sqrt(mean_squared_error(y_test, y_pred_reg_km))
        print(f"   Per-KMeans-Regime Test RMSE (using best features): {rmse_reg_km:.6f}")

        # 10a) Ensemble average (KMeans)
        print("\n10a) Ensemble average of best global and per-KMeans-regime models...")
        y_pred_ens_km = 0.5 * (y_pred_best_global + y_pred_reg_km)
        rmse_ens_km = np.sqrt(mean_squared_error(y_test, y_pred_ens_km))
        print(f"    Ensemble (KMeans-Regime) Test RMSE: {rmse_ens_km:.6f}")
    else:
        print("   Skipping per-KMeans-Regime RMSE and Ensemble: No valid per-KMeans-regime predictions made.")
        rmse_reg_km = np.nan
        rmse_ens_km = np.nan 
else:
    print("   Skipping per-KMeans-Regime models, RMSE, and Ensemble: No per-KMeans-regime models were trained.")
    rmse_reg_km = np.nan
    rmse_ens_km = np.nan

# 10b) Train Per-GMM-Regime Models
print("\n10b) Training per-GMM-regime models using best feature set...")
models_reg_gmm = {}
reg_dev_gmm = df_dev_clean["vol_state"].values # Integer GMM states, should be clean of NaNs

for state_val in np.unique(reg_dev_gmm): # state_val will be integers 0, 1, ...
    mask = (reg_dev_gmm == state_val)
    if np.sum(mask) > 0:
        m = xgb.XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            **best_params
        )
        m.fit(X_dev_best_features[mask], y_dev[mask])
        models_reg_gmm[state_val] = m
    else: # Should ideally not be hit
        print(f"   Skipping GMM regime {state_val} for per-regime model training: no data in dev set.")

reg_test_gmm = df_test_clean["vol_state"].values # Integer GMM states, should be clean
y_pred_reg_gmm = np.zeros_like(y_test, dtype=float)
valid_predictions_count_gmm = 0
per_gmm_regime_models_trained = bool(models_reg_gmm)

if per_gmm_regime_models_trained:
    for i, r_test_gmm in enumerate(reg_test_gmm): # r_test_gmm are integers
        if r_test_gmm in models_reg_gmm:
            y_pred_reg_gmm[i] = models_reg_gmm[r_test_gmm].predict(X_test_best_features[i].reshape(1, -1))[0]
            valid_predictions_count_gmm +=1
        else:
            print(f"   Warning: GMM Regime {r_test_gmm} in test set not found in trained per-GMM-regime models. Using global prediction for instance {i}.")
            y_pred_reg_gmm[i] = y_pred_best_global[i] 

    if valid_predictions_count_gmm > 0:
        rmse_reg_gmm = np.sqrt(mean_squared_error(y_test, y_pred_reg_gmm))
        print(f"   Per-GMM-Regime Test RMSE (using best features): {rmse_reg_gmm:.6f}")

        # 10c) Ensemble average (GMM)
        print("\n10c) Ensemble average of best global and per-GMM-regime models...")
        y_pred_ens_gmm = 0.5 * (y_pred_best_global + y_pred_reg_gmm)
        rmse_ens_gmm = np.sqrt(mean_squared_error(y_test, y_pred_ens_gmm))
        print(f"    Ensemble (GMM-Regime) Test RMSE: {rmse_ens_gmm:.6f}")
    else:
        print("   Skipping per-GMM-Regime RMSE and Ensemble: No valid per-GMM-regime predictions made.")
        rmse_reg_gmm = np.nan 
        rmse_ens_gmm = np.nan
else:
    print("   Skipping per-GMM-Regime models, RMSE, and Ensemble: No per-GMM-regime models were trained.")
    rmse_reg_gmm = np.nan
    rmse_ens_gmm = np.nan

# 11) Feature‐importance plot
# This section remains the same
print("\n11) Generating feature importance plot for the best global model...")
try:
    importance_dict = best_model_overall.get_booster().get_score(importance_type="gain")
    if importance_dict: 
        importances = {best_feature_list[int(f[1:])]: v for f, v in importance_dict.items()}
        imp_series = pd.Series(importances).sort_values(ascending=False).head(10)
        fig, ax = plt.subplots(figsize=(10,7))
        imp_series.plot(kind="barh", ax=ax)
        ax.invert_yaxis()
        ax.set_title(f"Top 10 Features (Best Global Model: '{best_scenario_name}')")
        plt.tight_layout()
        plt.savefig(f"feature_importance_{best_scenario_name.replace(' ', '_').lower()}.png")
        print(f"    Feature importance plot saved to feature_importance_{best_scenario_name.replace(' ', '_').lower()}.png")
    else:
        print("    Could not generate feature importances (model might be trivial or features had no gain).")
except Exception as e:
    print(f"    Error generating feature importance plot: {e}")

# 12) Additional diagnostics
print("\n12) Additional diagnostics for the best global model...")
# 12a) Tail performance
# This section remains the same
thresh = np.percentile(y_test, 90)
mask_tail = y_test >= thresh
if np.sum(mask_tail) > 0:
    tail_rmse = np.sqrt(mean_squared_error(y_test[mask_tail], y_pred_best_global[mask_tail]))
    print(f"12a) Hold-out Tail RMSE (90th pct, best global model): {tail_rmse:.6f}")
else:
    print("12a) Not enough samples in the tail (90th pct) to calculate Tail RMSE.")

# 12b) KMeans Regime‐specific Test RMSE
# This section uses your improved logic
print("12b) KMeans Regime-specific Test RMSE (for the best global model predictions):")
reg_test_km_values = df_test_clean["vol_cluster"].values 
unique_km_regimes_in_test = pd.Series(reg_test_km_values).dropna().unique()

if len(unique_km_regimes_in_test) > 0:
    for state in unique_km_regimes_in_test:
        idx = (reg_test_km_values == state)
        rmse_state = np.sqrt(mean_squared_error(y_test[idx], y_pred_best_global[idx]))
        print(f"     KMeans Regime '{state}': {rmse_state:.6f} (n={np.sum(idx)})")
else:
    print("     No valid (non-NaN) KMeans regimes found in the test set to calculate regime-specific RMSE.")

# 12c) GMM Regime‐Specific Test RMSE
print("12c) GMM Regime-specific Test RMSE (for the best global model predictions):")
reg_test_gmm_values = df_test_clean["vol_state"].values # Integer GMM states, should be clean of NaNs

# Get unique non-NaN GMM regime states present in the test set
unique_gmm_regimes_in_test = pd.Series(reg_test_gmm_values).dropna().unique() 

if len(unique_gmm_regimes_in_test) > 0:
    for state_val in unique_gmm_regimes_in_test:
        idx = (reg_test_gmm_values == state_val)
        rmse_state_gmm = np.sqrt(mean_squared_error(y_test[idx], y_pred_best_global[idx]))
        print(f"     GMM Regime {int(state_val)}: {rmse_state_gmm:.6f} (n={np.sum(idx)})") # Ensure state_val is int for print
else:
    print("     No valid (non-NaN) GMM regimes found in the test set to calculate regime-specific RMSE.")


# 13) Diebold–Mariano test
# This section remains the same
print("\n13) Diebold–Mariano test (best global model vs. naive baseline)...")
def diebold_mariano(loss1, loss2):
    """
    Compares two sets of forecast losses using the Diebold-Mariano test.
    Helps determine if one forecast is significantly better than another.
    """
    d = loss1 - loss2
    T = len(d)
    if T == 0:
        return np.nan, np.nan, "Not enough data for DM test."
    
    mean_d = np.mean(d)
    var_d = np.var(d, ddof=1) 
    
    if var_d == 0:
        if mean_d == 0:
            return 0.0, 1.0, "DM Test: Zero variance in differences and zero mean difference."
        else: 
            dm_stat_val = np.sign(mean_d) * np.inf 
            pval = 0.0
            return dm_stat_val, pval, "DM Test: Zero variance in differences, non-zero mean difference."

    dm_stat_val = mean_d / np.sqrt(var_d / T)
    pval = 2 * stats.t.sf(abs(dm_stat_val), df=T - 1) 
    return dm_stat_val, pval, ""

se_model = (y_test - y_pred_best_global)**2
naive_forecast = df_test_clean["daily_return"].abs().values

if len(naive_forecast) == len(y_test):
    se_naive = (y_test - naive_forecast)**2
    dm_stat, dm_p, dm_msg = diebold_mariano(se_model, se_naive)
    
    if dm_msg:
        print(f"    {dm_msg}")
    
    if pd.notna(dm_stat) and pd.notna(dm_p):
        print(f"    Diebold-Mariano DM stat: {dm_stat:.3f}, p-value: {dm_p:.3f}")
        if dm_p < 0.05:
            if dm_stat < 0: 
                print("    Interpretation: Best model is significantly better than naive baseline.")
            elif dm_stat > 0: 
                print("    Interpretation: Naive baseline is significantly better than best model.")
            else: 
                 print("    Interpretation: No significant difference detected (DM stat is effectively 0).")
        else:
            print("    Interpretation: No significant difference detected between best model and naive baseline (p >= 0.05).")
    elif not dm_msg: 
        print(f"    Diebold-Mariano test could not be computed reliably.")
else:
    print("    Could not run Diebold-Mariano test: length mismatch between naive forecast and y_test (should not happen with current data prep).")

print("\nDone.")