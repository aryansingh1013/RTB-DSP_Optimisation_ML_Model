import polars as pl
import numpy as np
import sys
import os

# Add dashboard to path to import utils
sys.path.append(os.path.join(os.getcwd(), 'dashboard'))
from utils import InferenceEngine

# Mock config paths
CTR_MODEL = "ctr_model_calibrated.pkl"
CVR_MODEL = "cvr_model_calibrated.pkl"
HASHER = "feature_hasher.pkl"
HASHER_CVR = "feature_hasher_cvr.pkl"
CONFIG = "feature_config.pkl"
CONFIG_CVR = "feature_config_cvr.pkl"
DATA = "merged_training_data.parquet"

def safe_path(p):
    return os.path.abspath(p)

print("Loading data...")
try:
    df = pl.read_parquet(DATA).sample(n=1000, seed=42)
except Exception as e:
    print(f"Error loading data: {e}")
    sys.exit(1)

print("Initializing engine...")
try:
    engine = InferenceEngine(
        safe_path(CTR_MODEL), safe_path(CVR_MODEL),
        safe_path(HASHER), safe_path(HASHER_CVR),
        safe_path(CONFIG), safe_path(CONFIG_CVR)
    )
except Exception as e:
    print(f"Error init engine: {e}")
    sys.exit(1)

print("Predicting...")
ctrs = []
cvrs = []

for i in range(len(df)):
    row = df.row(i, named=True)
    p_ctr, p_cvr = engine.predict(row)
    ctrs.append(p_ctr)
    cvrs.append(p_cvr)

ctrs = np.array(ctrs)
cvrs = np.array(cvrs)

print("\n--- Statistics ---")
print(f"pCTR Mean: {ctrs.mean():.6f}")
print(f"pCTR Median: {np.median(ctrs):.6f}")
print(f"pCTR Max: {ctrs.max():.6f}")
print("-" * 20)
print(f"pCVR Mean: {cvrs.mean():.6f}")
print(f"pCVR Median: {np.median(cvrs):.6f}")
print(f"pCVR Max: {cvrs.max():.6f}")
