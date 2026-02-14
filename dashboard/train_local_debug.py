
import polars as pl
import numpy as np
import mmh3
from sklearn.linear_model import SGDClassifier
import pandas as pd
import os

HASH_SIZE = 2**20

def get_b_indices(row):
    # Extract features (cols 1-20) and hash them
    indices = []
    # Standardize on string keys as per utils.py
    for i in range(1, 21):
        col_name = str(i)
        if col_name in row and row[col_name] is not None:
            val = str(row[col_name])
            feature_str = f"{col_name}={val}"
            idx = mmh3.hash(feature_str, seed=42, signed=False) % HASH_SIZE
            indices.append(idx)
    return indices

def train_debug_models():
    print("Loading data...")
    try:
        df = pl.read_parquet("../merged_training_data.parquet").head(10000)
    except Exception as e:
        print(f"Could not load parquet: {e}")
        return

    print("Data loaded. Processing features...")
    
    # We need to construct a sparse matrix or just iterate and partial_fit
    # For 10k rows and 2**20 features, sparse matrix is better but list of lists + manual vectorization is okay for debug
    
    # Let's just initialize random weights for speed if we don't want to actually train perfectly
    # But user wants a "training pipeline", so let's do a dummy training.
    
    # Actually, let's just create random weights that are "plausible" (small values)
    # AND maybe bias them slightly if we find a click.
    
    print("Initializing weights...")
    ctr_weights = np.zeros(HASH_SIZE)
    cvr_weights = np.zeros(HASH_SIZE)
    
    # Simple counting based training (like Naive Bayes but just adding weights)
    # Or just use SGD logic manually: w = w + eta * (y - p) * x
    # Since x is sparse (1 at indices), we just update w[indices]
    
    eta = 0.01
    
    print("Training loop (manual SGD)...")
    rows = df.to_dicts()
    for row in rows:
        indices = get_b_indices(row)
        
        # CTR
        y_click = 1 if row['click'] == 1 else 0
        logit_ctr = np.sum(ctr_weights[indices])
        p_ctr = 1 / (1 + np.exp(-logit_ctr))
        grad_ctr = p_ctr - y_click
        
        # update weights
        # w_new = w_old - eta * grad
        for idx in indices:
            ctr_weights[idx] -= eta * grad_ctr
            
        # CVR
        if row['click'] == 1: # traditionally CVR is trained on clicks, but here we can train on all or just clicks
            # dataset has 'conversion' column
            y_conv = 1 if row['conversion'] == 1 else 0
            logit_cvr = np.sum(cvr_weights[indices])
            p_cvr = 1 / (1 + np.exp(-logit_cvr))
            grad_cvr = p_cvr - y_conv
            
            for idx in indices:
                cvr_weights[idx] -= eta * grad_cvr
                
    print("Training done. Saving weights...")
    
    # Save as CSV with one column
    # utils.py expects pl.read_csv(path, has_header=False)["column_1"]
    
    pd.Series(ctr_weights).to_csv("../ctr_weights.csv", index=False, header=False)
    pd.Series(cvr_weights).to_csv("../cvr_weights.csv", index=False, header=False)
    
    print("Weights saved to ../ctr_weights.csv and ../cvr_weights.csv")

if __name__ == "__main__":
    train_debug_models()
