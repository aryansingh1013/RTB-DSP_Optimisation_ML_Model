# =============================================================================
# CTR PREDICTION MODEL - IMPROVED VERSION
# Adapted for generic features '1'..'20'
# =============================================================================

import polars as pl
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    roc_auc_score, 
    log_loss, 
    precision_score, 
    recall_score,
    confusion_matrix
)
import joblib
import time
import os
import gc
import warnings

warnings.filterwarnings('ignore')

print("="*70)
print("CTR PREDICTION MODEL - IMPROVED VERSION")
print("="*70)

# Configuration
DATA_PATH = "../merged_training_data.parquet"
MODEL_FILENAME = "../ctr_model_calibrated.pkl"
HASHER_FILENAME = "../feature_hasher.pkl"
CONFIG_FILENAME = "../feature_config.pkl"

# =============================================================================
# STEP 1: DATA LOADING & PREPARATION
# =============================================================================
print("\n" + "="*70)
print("STEP 1: DATA LOADING & PREPARATION")
print("="*70)

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pl.read_parquet(DATA_PATH)
        
        # Cast columns to appropriate types if needed
        feature_cols = [str(i) for i in range(1, 21)]
        df = df.with_columns([
            pl.col(c).cast(pl.Utf8) for c in feature_cols
        ])
        
        print(f"Loaded {df.height} rows.")
        return df
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return None

df = load_data()
if df is None:
    raise ValueError("Could not load data.")

# Target
target_column = 'click'
feature_columns = [str(i) for i in range(1, 21)]

# =============================================================================
# STEP 2: FEATURE HASHING
# =============================================================================
print("\n" + "="*70)
print("STEP 2: FEATURE HASHING")
print("="*70)

def convert_row_to_dict(row, feature_cols):
    result = {}
    for col in feature_cols:
        if col in row and row[col] is not None:
            val = str(row[col])
            # Filter out 'None', 'nan', 'null' strings just in case
            if val.lower() not in ['none', 'nan', 'null', '']:
                result[f"{col}={val}"] = 1.0 # FeatureHasher expects numeric values for values
    return result

print("Converting rows to dictionary format...")
# X_raw is already a list of dicts from polars.to_dicts()
# We need to process it to match the format FeatureHasher expects: 
# [{feature_name: value, ...}, ...] where value is numeric (usually 1.0 for categorical)
# The current X_raw from polars contains {col: value} where value is string/categorical.
# FeatureHasher with input_type='dict' expects dicts like {'feat=val': 1.0} for categorical features
# OR {'feat': val} if val is numeric. 
# Our columns are categorical, so we need to transform {col: val} -> {col=val: 1.0}

print("Transforming dictionaries for FeatureHasher...")
X_dict_list = []
rows = df.select(feature_columns).to_dicts()

for row in rows:
    X_dict_list.append(convert_row_to_dict(row, feature_columns))

print(f"Captured {len(X_dict_list):,} rows")


# Define y before we delete df
y = df.select(target_column).to_numpy().flatten()

# Feature Hashing
N_FEATURES = 2**20  # 1 million features for collision avoidance
print(f"   • n_features: {N_FEATURES:,}")

hasher = FeatureHasher(n_features=N_FEATURES, input_type='dict', alternate_sign=True)

print("Applying feature hashing...")
X_hashed = hasher.fit_transform(X_dict_list)

print(f"Feature Hashing Complete!")
print(f"   • Output shape: {X_hashed.shape}")

# Clean up
del df, X_dict_list
gc.collect()

# =============================================================================
# STEP 3: THREE-WAY SPLIT
# =============================================================================
print("\n" + "="*70)
print("STEP 3: THREE-WAY SPLIT")
print("="*70)

# First split: 80% train+calib, 20% validation
X_temp, X_val, y_temp, y_val = train_test_split(
    X_hashed, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Second split: 75% train, 25% calibration (of the 80%) -> 60% train, 20% calib total
X_train, X_calib, y_train, y_calib = train_test_split(
    X_temp, y_temp,
    test_size=0.25,
    random_state=42,
    stratify=y_temp
)

print(f"Training Set:    {X_train.shape[0]:,} samples, {y_train.sum():,} clicks")
print(f"Calibration Set: {X_calib.shape[0]:,} samples, {y_calib.sum():,} clicks")
print(f"Validation Set:  {X_val.shape[0]:,} samples, {y_val.sum():,} clicks")

# =============================================================================
# STEP 4: TRAIN BASE MODEL
# =============================================================================
print("\n" + "="*70)
print("STEP 4: TRAIN BASE LOGISTIC REGRESSION MODEL")
print("="*70)

base_model = LogisticRegression(
    solver='saga',
    max_iter=100, # Reduced from 200 for speed on large dataset, usually enough
    class_weight='balanced',
    n_jobs=-1,
    C=0.1,
    random_state=42,
    verbose=1,
    tol=1e-3
)

print("Training base model...")
start_time = time.time()
base_model.fit(X_train, y_train)
print(f"Base model training complete in {time.time() - start_time:.2f} seconds")

# Base Evaluation
y_pred_base = base_model.predict_proba(X_val)[:, 1]
print(f"   • Base Model AUC: {roc_auc_score(y_val, y_pred_base):.4f}")
print(f"   • Base Mean pCTR: {y_pred_base.mean():.6f} (vs Actual {y_val.mean():.6f})")

# =============================================================================
# STEP 5: CALIBRATION
# =============================================================================
print("\n" + "="*70)
print("STEP 5: PROBABILITY CALIBRATION")
print("="*70)

calibrated_model = CalibratedClassifierCV(
    base_model,
    method='isotonic',
    cv='prefit'
)

print("Fitting calibration model...")
calibrated_model.fit(X_calib, y_calib)
print("Calibration complete!")

# =============================================================================
# STEP 6: EVALUATION
# =============================================================================
print("\n" + "="*70)
print("STEP 6: EVALUATION")
print("="*70)

y_pred_proba = calibrated_model.predict_proba(X_val)[:, 1]
roc_auc = roc_auc_score(y_val, y_pred_proba)
logloss = log_loss(y_val, y_pred_proba)

print(f"CALIBRATED METRICS:")
print(f"   • ROC-AUC:  {roc_auc:.6f}")
print(f"   • Log Loss: {logloss:.6f}")
print(f"   • Mean pCTR: {y_pred_proba.mean():.6f} (vs Actual {y_val.mean():.6f})")
print(f"   • Ratio: {y_pred_proba.mean() / y_val.mean():.4f}x")

# =============================================================================
# STEP 8: SAVE
# =============================================================================
print("\n" + "="*70)
print("STEP 8: SAVE")
print("="*70)

joblib.dump(calibrated_model, MODEL_FILENAME)
joblib.dump(hasher, HASHER_FILENAME)

feature_config = {
    'feature_columns': feature_columns,
    'n_features': N_FEATURES,
    'metrics': {'auc': roc_auc, 'logloss': logloss}
}
joblib.dump(feature_config, CONFIG_FILENAME)

print(f"Saved models to {os.path.abspath(MODEL_FILENAME)}")
print(f"Saved hasher to {os.path.abspath(HASHER_FILENAME)}")
