import polars as pl
import numpy as np
import mmh3
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score
import os
import time
from scipy.sparse import csr_matrix

# Configuration
DATA_PATH = "../merged_training_data.parquet"
HASH_SIZE = 2**20  # ~1 million features

def load_data():
    print(f"Loading data from {DATA_PATH}...")
    try:
        df = pl.read_parquet(DATA_PATH)
        
        # Cast columns to appropriate types if needed
        # Assuming 1-20 are strings/categoricals
        feature_cols = [str(i) for i in range(1, 21)]
        df = df.with_columns([
            pl.col(c).cast(pl.Utf8) for c in feature_cols
        ])
        
        print(f"Loaded {df.height} rows.")
        return df
    except Exception as e:
        print(f"Error loading parquet: {e}")
        # Fallback to synthetic if file not found
        return generate_synthetic_data(10000)

def generate_synthetic_data(n_samples):
    print("Generating synthetic data...")
    np.random.seed(42)
    data = {
        "click": np.random.randint(0, 2, n_samples),
        "conversion": np.random.randint(0, 2, n_samples),
        "payprice": np.random.randint(10, 100, n_samples),
    }
    # Generate dummy features 1-20
    for i in range(1, 21):
        data[str(i)] = [f"feat_{i}_{x}" for x in np.random.randint(0, 100, n_samples)]
    
    data["conversion"] = data["click"] * data["conversion"]
    return pl.DataFrame(data)

def get_hashed_index(feature_name, value):
    feature_str = f"{feature_name}={value}"
    return mmh3.hash(feature_str, seed=42, signed=False) % HASH_SIZE

def transform_batch(batch_df):
    X_indices = []
    X_values = []
    y_click = []
    y_conversion = []
    
    # Select columns present in Parquet dataset
    # Features are named '1' to '20'
    cat_cols = [str(i) for i in range(1, 21)]
    
    # Check if cols exist
    available_cols = [c for c in cat_cols if c in batch_df.columns]
    
    rows = batch_df.to_dicts()
    for row in rows:
        indices = []
        
        # 1. Single Features
        current_indices = []
        for col in available_cols:
            if row[col] is not None:
                idx = get_hashed_index(col, row[col])
                current_indices.append(idx)
        
        # 2. Cross Features (Example: Feature 1 x Feature 2)
        # Using cols "1" and "2" if they exist
        if "1" in batch_df.columns and "2" in batch_df.columns:
             cross_val = f"{row.get('1', '')}_x_{row.get('2', '')}"
             current_indices.append(get_hashed_index("1_x_2", cross_val))
        
        X_indices.append(current_indices)
        X_values.append([1.0] * len(current_indices))
        
        # Labels
        y_click.append(row["click"])
        y_conversion.append(row["conversion"])
        
    return np.array(X_indices, dtype=object), np.array(X_values, dtype=object), np.array(y_click), np.array(y_conversion)

def train_and_evaluate(df):
    # Split data for evaluation (80% train, 20% validation)
    # Since it's time-series data usually, we should split by time, but here random split for simplicity
    # or just use the last chunk.
    
    n_total = df.height
    n_train = int(n_total * 0.8)
    
    df_train = df.slice(0, n_train)
    df_val = df.slice(n_train, n_total - n_train)
    
    print(f"Splitting data: {n_train} training samples, {n_total - n_train} validation samples.")

    # Initialize models
    ctr_model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, fit_intercept=False, learning_rate='optimal', random_state=42)
    cvr_model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, fit_intercept=False, learning_rate='optimal', random_state=42)

    # --- Training ---
    print("Transforming training data...")
    X_idx, X_val, y_clk, y_cnv = transform_batch(df_train)
    
    # CSR Matrix Construction
    n_samples = len(y_clk)
    n_features_per_sample = [len(x) for x in X_idx]
    
    row_indices = np.repeat(np.arange(n_samples), n_features_per_sample)
    col_indices = np.concatenate(X_idx)
    data = np.concatenate(X_val)
    
    X_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, HASH_SIZE))
    
    # CTR Training (with downsampling)
    DOWNSAMPLE_RATE = 0.1
    pos_mask = y_clk == 1
    neg_mask = (y_clk == 0) & (np.random.rand(n_samples) < DOWNSAMPLE_RATE)
    train_mask = pos_mask | neg_mask
    
    X_train_ctr = X_sparse[train_mask]
    y_train_ctr = y_clk[train_mask]
    
    print(f"Training CTR model on {X_train_ctr.shape[0]} samples...")
    if X_train_ctr.shape[0] > 0:
        ctr_model.partial_fit(X_train_ctr, y_train_ctr, classes=[0, 1])
        
    # CVR Training (clicked only)
    clicked_mask = y_clk == 1
    X_train_cvr = X_sparse[clicked_mask]
    y_train_cvr = y_cnv[clicked_mask]
    
    print(f"Training CVR model on {X_train_cvr.shape[0]} samples...")
    if X_train_cvr.shape[0] > 0:
        cvr_model.partial_fit(X_train_cvr, y_train_cvr, classes=[0, 1])
        
    # --- Evaluation ---
    print("\nEvaluating on Validation Set...")
    X_idx_val, X_val_val, y_clk_val, y_cnv_val = transform_batch(df_val)
    
    n_val = len(y_clk_val)
    n_feat_val = [len(x) for x in X_idx_val]
    row_idx_val = np.repeat(np.arange(n_val), n_feat_val)
    col_idx_val = np.concatenate(X_idx_val)
    data_val = np.concatenate(X_val_val)
    
    X_val_sparse = csr_matrix((data_val, (row_idx_val, col_idx_val)), shape=(n_val, HASH_SIZE))
    
    # CTR Evaluation
    if n_val > 0:
        probs_ctr = ctr_model.predict_proba(X_val_sparse)[:, 1]
        preds_ctr = (probs_ctr > 0.5).astype(int)
        
        auc_ctr = roc_auc_score(y_clk_val, probs_ctr) if len(np.unique(y_clk_val)) > 1 else 0
        logloss_ctr = log_loss(y_clk_val, probs_ctr) if len(np.unique(y_clk_val)) > 1 else 0
        acc_ctr = accuracy_score(y_clk_val, preds_ctr)
        
        print(f"CTR Model Metrics:")
        print(f"  Accuracy: {acc_ctr:.4f}")
        print(f"  AUC:      {auc_ctr:.4f}")
        print(f"  Log Loss: {logloss_ctr:.4f}")
        
    # CVR Evaluation (on clicks in validation info)
    # We can evaluate CVR on all validation data or just clicks. Usually just clicks for CVR.
    # But for the purpose of "CVR Model Accuracy", let's look at clicks only.
    val_clicks_mask = y_clk_val == 1
    if np.sum(val_clicks_mask) > 0:
        X_val_cvr = X_val_sparse[val_clicks_mask]
        y_val_cvr_sub = y_cnv_val[val_clicks_mask]
        
        if X_val_cvr.shape[0] > 0:
            probs_cvr = cvr_model.predict_proba(X_val_cvr)[:, 1]
            preds_cvr = (probs_cvr > 0.5).astype(int)
            
            auc_cvr = roc_auc_score(y_val_cvr_sub, probs_cvr) if len(np.unique(y_val_cvr_sub)) > 1 else 0
            acc_cvr = accuracy_score(y_val_cvr_sub, preds_cvr)
            
            print(f"CVR Model Metrics (on clicked impressions):")
            print(f"  Accuracy: {acc_cvr:.4f}")
            print(f"  AUC:      {auc_cvr:.4f}")
            
            # Additional stats for debugging
            print(f"  pCVR Stats: Min={probs_cvr.min():.4f}, Max={probs_cvr.max():.4f}, Mean={probs_cvr.mean():.4f}")
            print(f"  Sample pCVR: {probs_cvr[:10]}")
            print(f"  Sample Labels: {y_val_cvr_sub[:10]}")
    else:
        print("No clicks in validation set to evaluate CVR.")

    return ctr_model, cvr_model

def export_weights(model, filename):
    if not hasattr(model, "coef_"):
        model.partial_fit(csr_matrix((1, HASH_SIZE)), [0], classes=[0, 1])
        
    weights = model.coef_[0]
    
    # Create a DataFrame for fast export
    df_weights = pl.DataFrame({
        "weight": weights
    })
    
    # Export without header
    df_weights.write_csv(filename, include_header=False)
    print(f"Exported {filename} with shape {df_weights.shape}")

if __name__ == "__main__":
    df = load_data()
    ctr_model, cvr_model = train_and_evaluate(df)
    
    # Export weights to parent directory for dashboard to pick up
    export_weights(ctr_model, "../ctr_weights.csv")
    export_weights(cvr_model, "../cvr_weights.csv")
