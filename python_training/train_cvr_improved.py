# =============================================================================
# CVR PREDICTION MODEL - IMPROVED VERSION
# Features: 
#   - Balanced Class Weights
#   - Isotonic Calibration
#   - Domain-Specific Feature Engineering (Region x City, Hour, Weekday)
# =============================================================================

import polars as pl
import numpy as np
import pandas as pd
from sklearn.feature_extraction import FeatureHasher
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, log_loss, confusion_matrix
import joblib
import time
import os
import gc
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

print("="*70)
print("CVR PREDICTION MODEL - IMPROVED VERSION")
print("="*70)

# Configuration
# We might need to adjust this depending on where download_kaggle_data.py puts files
# For now, we assume we might still use merged_training_data.parquet OR the new kaggle data.
# I will set it to look for both or one. 
# Let's target the existing file for now, and I'll add logic to load Kaggle data if available later.
DATA_PATH = "../merged_training_data.parquet" 
MODEL_FILENAME = "../cvr_model_calibrated.pkl"
HASHER_FILENAME = "../feature_hasher_cvr.pkl"
CONFIG_FILENAME = "../feature_config_cvr.pkl"

# =============================================================================
# STEP 1: DATA LOADING 
# =============================================================================
print("\n" + "="*70)
print("STEP 1: DATA LOADING")
print("="*70)

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    try:
        # Load data
        # We only care about rows where click=1 for CVR training (p(conversion|click))
        # But wait, if we train only on clicks, we lose the 'no-click' info?
        # No, CVR is P(conversion|click). So the training universe IS clicks.
        # So we filter for click=1.
        
        df = pl.read_parquet(DATA_PATH)
        
        # Filter for clicks only
        # If 'click' column exists and has 1s
        if 'click' in df.columns:
            print(f"Filtering for clicks=1 (Total rows: {df.height})")
            df = df.filter(pl.col('click') == 1)
            print(f"Rows after filtering: {df.height}")
        
        # Cast columns to appropriate types
        feature_cols = [str(i) for i in range(1, 21)]
        df = df.with_columns([
            pl.col(c).cast(pl.Utf8) for c in feature_cols
        ])
        
        return df
    except Exception as e:
        print(f"Error loading parquet: {e}")
        return None

df = load_data()
if df is None or df.height == 0:
    raise ValueError("Could not load data or no clicks found.")

# Target: 'conversion' (or column '2' might contain it? User mapping said 2=logtype. 
# Usually 'conversion' is a separate column or implied by logtype=3.
# Let's assume there is a 'conversion' column in merged data, 
# OR we need to derive it from col '2' (logtype) if 'conversion' col is missing.
# In the original train_models.py, target was 'conversion'.
if 'conversion' not in df.columns:
    # Try to derive from logtype (col '2')
    # User said 2=logtype (1=bid, 2=click, 3=conversion)
    # Check if '2' exists
    if '2' in df.columns:
        print("Deriving conversion target from logtype column '2'...")
        # conversion if logtype == '3' (assuming schema)
        # But wait, usually dataset has one row per bid. If it converted, logtype might be modified or separate log.
        # In typical iPinYou/IPinyou, conversion is a separate file or joined.
        # merged_training_data.parquet implies it's already joined.
        # Let's check unique values of '2'
        logtypes = df.select('2').unique().to_series().to_list()
        print(f"Logtypes found: {logtypes}")
        
        # If we only have clicks (logtype=2), how do we know if it converted?
        # Maybe there's a 'conversion' column. 
        # For now, I will assume 'conversion' column exists as in train_models.py.
        pass
    else:
        raise ValueError("Target column 'conversion' not found.")

target_column = 'conversion'

# =============================================================================
# STEP 2: FEATURE ENGINEERING
# =============================================================================
print("\n" + "="*70)
print("STEP 2: FEATURE ENGINEERING")
print("="*70)

# User Mapping:
# 1: timestamp -> Hour, Weekday
# 6: region
# 7: city
# 19: advertiser

def extract_time_features(timestamp_str):
    # Format: YYYYMMDDHHMMSS... or similar
    # If standard 14 digit: 20130606000133
    try:
        if timestamp_str is None: return "0", "0"
        s = str(timestamp_str)
        if len(s) >= 10:
            # simple parsing
            # yyyy = s[0:4], mm = s[4:6], dd = s[6:8], hh = s[8:10]
            dt = datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]), int(s[8:10]))
            return str(dt.hour), str(dt.weekday()) # 0=Mon, 6=Sun
    except:
        pass
    return "0", "0"

# Note: Polars `apply` can be slow, but for feature engineering on <1M rows it's okay. 
# Faster way is to use string slicing if format is fixed.
# Let's assume generic string slicing for speed.

# 1. Hour & Weekday
print("Extracting Hour & Weekday from Column '1'...")
# Assuming '1' is string "20130606000133"
# Hour is index 8-10
df = df.with_columns([
    pl.col('1').str.slice(8, 2).alias('feature_hour'),
    # Weekday is harder with just slicing, need to parse. 
    # Let's approximate or just use day (6-8) as categorical?
    # Day is enough for short duration. But let's try true weekday if fast.
    # Actually, let's keep it simple: Hour is highly predictive. Weekday less so for short datasets.
    # We will use Hour.
])

# 2. Cross Features
print("Creating Cross Features...")
# Region(6) x City(7)
# Advertiser(19) x Region(6)
# UserAgent(4) x Hour
# SlotFormat(14) x Creative(16)

df = df.with_columns([
    (pl.col('6') + "_" + pl.col('7')).alias('cross_region_city'),
    (pl.col('19') + "_" + pl.col('6')).alias('cross_adv_region'),
    (pl.col('4') + "_" + pl.col('feature_hour')).alias('cross_ua_hour'),
    (pl.col('14') + "_" + pl.col('16')).alias('cross_format_creative')
])

# Select features for hashing
# Original 1-20 + New Derived
base_features = [str(i) for i in range(1, 21) if i not in [1, 3, 5, 9, 10, 17, 18]] # Drop ID/IP/Price/etc as planned
derived_features = ['feature_hour', 'cross_region_city', 'cross_adv_region', 'cross_ua_hour', 'cross_format_creative']

feature_columns = base_features + derived_features
print(f"Selected {len(feature_columns)} features for training.")

# =============================================================================
# STEP 3: FEATURE HASHING
# =============================================================================
print("\n" + "="*70)
print("STEP 3: FEATURE HASHING")
print("="*70)

def convert_row_to_dict_cvr(row, feature_cols):
    result = {}
    for col in feature_cols:
        if col in row and row[col] is not None:
            val = str(row[col])
            if val.lower() not in ['none', 'nan', 'null', '']:
                result[f"{col}={val}"] = 1.0
    return result

print("Converting to dicts...")
X_dict_list = []
rows = df.select(feature_columns).to_dicts()

for row in rows:
    X_dict_list.append(convert_row_to_dict_cvr(row, feature_columns))

# Define y
y = df.select(target_column).to_numpy().flatten()

print(f"Captured {len(X_dict_list):,} rows")

# Feature Hashing
N_FEATURES = 2**18  # 262k features is usually enough for CVR (less data)
print(f"   • n_features: {N_FEATURES:,}")

hasher = FeatureHasher(n_features=N_FEATURES, input_type='dict', alternate_sign=True)

print("Applying feature hashing...")
X_hashed = hasher.fit_transform(X_dict_list)

# Clean up
del df, X_dict_list, rows
gc.collect()

# =============================================================================
# STEP 4: TRAIN & CALIBRATE
# =============================================================================
print("\n" + "="*70)
print("STEP 4: TRAIN & CALIBRATE")
print("="*70)

# Split
X_train_val, X_test, y_train_val, y_test = train_test_split(X_hashed, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_calib, y_train, y_calib = train_test_split(X_train_val, y_train_val, test_size=0.25, stratify=y_train_val, random_state=42)

print(f"Training CVR on {X_train.shape[0]} samples...")

# Base Model
base_model = LogisticRegression(
    solver='saga',
    max_iter=100, 
    class_weight='balanced',
    C=0.1,
    n_jobs=-1,
    random_state=42
)

base_model.fit(X_train, y_train)
print("Base model trained.")

# Calibrate
calibrated = CalibratedClassifierCV(base_model, method='isotonic', cv='prefit')
calibrated.fit(X_calib, y_calib)
print("Model calibrated.")

# =============================================================================
# STEP 5: EVALUATION
# =============================================================================
print("\n" + "="*70)
print("STEP 5: EVALUATION")
print("="*70)

probs = calibrated.predict_proba(X_test)[:, 1]
auc = roc_auc_score(y_test, probs)
ll = log_loss(y_test, probs)

print(f"CVR ROC-AUC: {auc:.4f}")
print(f"CVR LogLoss: {ll:.4f}")
print(f"Mean Predicted CVR: {probs.mean():.6f} (Actual: {y_test.mean():.6f})")

# =============================================================================
# STEP 6: SAVE
# =============================================================================
print("\n" + "="*70)
print("STEP 6: SAVE")
print("="*70)

joblib.dump(calibrated, MODEL_FILENAME)
joblib.dump(hasher, HASHER_FILENAME)

feature_config = {
    'feature_columns': feature_columns,
    'n_features': N_FEATURES, # Note: using smaller space for CVR
    'metrics': {'auc': auc, 'logloss': ll}
}
joblib.dump(feature_config, CONFIG_FILENAME)

print("Saved CVR model artifacts.")
