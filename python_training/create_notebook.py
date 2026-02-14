import json
import os

# Helper for JSON null
null = None

notebook_content = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# RTB DSP Training Pipeline\n",
    "\n",
    "This notebook implements the training pipeline for the Real-Time Bidding (RTB) Demand-Side Platform (DSP) optimization engine.\n",
    "\n",
    "## Mdoel Architecture\n",
    "We use **Logistic Regression (SGDClassifier)** for both CTR and CVR prediction. \n",
    "- **CTR Model**: Predicts $P(click=1 | features)$. Trained on all impressions (with negative downsampling).\n",
    "- **CVR Model**: Predicts $P(conversion=1 | click=1, features)$. Trained only on clicked impressions (delayed feedback).\n",
    "\n",
    "While the *algorithm* is the same, the **weights** are different because they learn different probabilities training on different data slices.\n",
    "\n",
    "## Steps:\n",
    "1.  **Setup**: Install dependencies.\n",
    "2.  **Data Generation**: Create synthetic data.\n",
    "3.  **Preprocessing**: Clean and prepare data using Polars.\n",
    "4.  **Feature Engineering**: Apply Hashing Trick and Cross-Features.\n",
    "5.  **Model Training**: Train separate SGDClassifiers for CTR and CVR.\n",
    "6.  **Export**: Save weights to CSV for Java inference."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Install dependencies\n",
    "!pip install polars scikit-learn mmh3 numpy pandas"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import polars as pl\n",
    "import numpy as np\n",
    "import mmh3\n",
    "from sklearn.linear_model import SGDClassifier\n",
    "from sklearn.metrics import log_loss, roc_auc_score\n",
    "import os\n",
    "import time"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Data Loading (Parquet)\n",
    "\n",
    "We load the pre-merged dataset from `../merged_training_data.parquet`.\n",
    "The dataset contains:\n",
    "- **Features**: Columns `1` to `20` (Categorical features).\n",
    "- **Labels**: `click`, `conversion`.\n",
    "- **Metadata**: `payprice`, `win`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "DATA_PATH = \"../merged_training_data.parquet\"\n",
    "\n",
    "def load_data():\n",
    "    print(f\"Loading data from {DATA_PATH}...\")\n",
    "    try:\n",
    "        df = pl.read_parquet(DATA_PATH)\n",
    "        \n",
    "        # Cast columns to appropriate types if needed\n",
    "        # Assuming 1-20 are strings/categoricals\n",
    "        feature_cols = [str(i) for i in range(1, 21)]\n",
    "        df = df.with_columns([\n",
    "            pl.col(c).cast(pl.Utf8) for c in feature_cols\n",
    "        ])\n",
    "        \n",
    "        print(f\"Loaded {df.height} rows.\")\n",
    "        return df\n",
    "    except Exception as e:\n",
    "        print(f\"Error loading parquet: {e}\")\n",
    "        # Fallback to synthetic if file not found (for robust notebook)\n",
    "        return generate_synthetic_data(10000)\n",
    "\n",
    "# Fallback synthetic generator\n",
    "def generate_synthetic_data(n_samples):\n",
    "    np.random.seed(42)\n",
    "    data = {\n",
    "        \"click\": np.random.randint(0, 2, n_samples),\n",
    "        \"conversion\": np.random.randint(0, 2, n_samples),\n",
    "        \"payprice\": np.random.randint(10, 100, n_samples),\n",
    "    }\n",
    "    # Generate dummy features 1-20\n",
    "    for i in range(1, 21):\n",
    "        data[str(i)] = [f\"feat_{i}_{x}\" for x in np.random.randint(0, 100, n_samples)]\n",
    "    \n",
    "    data[\"conversion\"] = data[\"click\"] * data[\"conversion\"]\n",
    "    return pl.DataFrame(data)\n",
    "\n",
    "df = load_data()\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Preprocessing & Feature Engineering\n",
    "\n",
    "- **Hashing Trick**: Map categorical features to a fixed size vector ($2^{20}$).\n",
    "- **Cross-Features**: Combine features to capture interactions."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "HASH_SIZE = 2**20  # ~1 million features\n",
    "\n",
    "def get_hashed_index(feature_name, value):\n",
    "    feature_str = f\"{feature_name}={value}\"\n",
    "    return mmh3.hash(feature_str, seed=42, signed=False) % HASH_SIZE\n",
    "\n",
    "def transform_batch(batch_df):\n",
    "    X_indices = []\n",
    "    X_values = []\n",
    "    y_click = []\n",
    "    y_conversion = []\n",
    "    \n",
    "    # Select columns present in Parquet dataset\n",
    "    # Features are named '1' to '20'\n",
    "    cat_cols = [str(i) for i in range(1, 21)]\n",
    "    \n",
    "    # Check if cols exist\n",
    "    available_cols = [c for c in cat_cols if c in batch_df.columns]\n",
    "    \n",
    "    rows = batch_df.to_dicts()\n",
    "    for row in rows:\n",
    "        indices = []\n",
    "        \n",
    "        # 1. Single Features\n",
    "        for col in available_cols:\n",
    "            if row[col] is not None:\n",
    "                idx = get_hashed_index(col, row[col])\n",
    "                indices.append(idx)\n",
    "        \n",
    "        # 2. Cross Features (Example: Feature 1 x Feature 2)\n",
    "        # Assuming 1=Region, 2=City (just as example)\n",
    "        if \"1\" in batch_df.columns and \"2\" in batch_df.columns:\n",
    "             cross_val = f\"{row.get('1', '')}_x_{row.get('2', '')}\"\n",
    "             indices.append(get_hashed_index(\"1_x_2\", cross_val))\n",
    "        \n",
    "        X_indices.append(indices)\n",
    "        X_values.append([1.0] * len(indices))\n",
    "        \n",
    "        # Labels\n",
    "        y_click.append(row[\"click\"])\n",
    "        y_conversion.append(row[\"conversion\"])\n",
    "        \n",
    "    return np.array(X_indices, dtype=object), np.array(X_values, dtype=object), np.array(y_click), np.array(y_conversion)\n",
    "\n",
    "print(\"Transformation function defined.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Model Training (SGDClassifier)\n",
    "\n",
    "We use `SGDClassifier` with `loss='log_loss'` which implements Logistic Regression with Stochastic Gradient Descent.\n",
    "\n",
    "### Negative Downsampling\n",
    "To handle class imbalance and speed up training, we downsample the negative class (non-clicks). \n",
    "We keep 10% of negatives and all positives.\n",
    "\n",
    "**Important**: The Java inference engine must recalibrate the prediction using:\n",
    "$p_{calibrated} = \\frac{p}{p + (1-p)/w}$ where $w = 0.1$."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Initialize models\n",
    "# Two separate instances of SGDClassifier\n",
    "ctr_model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, fit_intercept=False, learning_rate='optimal', random_state=42)\n",
    "cvr_model = SGDClassifier(loss='log_loss', penalty='l2', alpha=0.0001, fit_intercept=False, learning_rate='optimal', random_state=42)\n",
    "\n",
    "from scipy.sparse import csr_matrix\n",
    "\n",
    "def train_models(df):\n",
    "    X_idx, X_val, y_clk, y_cnv = transform_batch(df)\n",
    "    \n",
    "    # Construct CSR Matrix for the whole batch first\n",
    "    n_samples = len(y_clk)\n",
    "    n_features_per_sample = X_idx.shape[1]\n",
    "    \n",
    "    row_indices = np.repeat(np.arange(n_samples), n_features_per_sample)\n",
    "    col_indices = X_idx.flatten()\n",
    "    data = X_val.flatten()\n",
    "    \n",
    "    X_sparse = csr_matrix((data, (row_indices, col_indices)), shape=(n_samples, HASH_SIZE))\n",
    "    \n",
    "    # --- CTR Training with Negative Downsampling ---\n",
    "    # Keep all positives (click=1)\n",
    "    # Keep 10% of negatives (click=0)\n",
    "    DOWNSAMPLE_RATE = 0.1\n",
    "    \n",
    "    pos_mask = y_clk == 1\n",
    "    # Randomly select 10% of negatives\n",
    "    neg_mask = (y_clk == 0) & (np.random.rand(n_samples) < DOWNSAMPLE_RATE)\n",
    "    \n",
    "    train_mask = pos_mask | neg_mask\n",
    "    \n",
    "    X_train_ctr = X_sparse[train_mask]\n",
    "    y_train_ctr = y_clk[train_mask]\n",
    "    \n",
    "    print(f\"Training CTR model on {X_train_ctr.shape[0]} samples (downsampled from {n_samples})\")\n",
    "    if X_train_ctr.shape[0] > 0:\n",
    "        ctr_model.partial_fit(X_train_ctr, y_train_ctr, classes=[0, 1])\n",
    "    \n",
    "    # --- CVR Training ---\n",
    "    # Train ONLY on clicks (P(Conversion | Click))\n",
    "    clicked_mask = y_clk == 1\n",
    "    \n",
    "    X_train_cvr = X_sparse[clicked_mask]\n",
    "    y_train_cvr = y_cnv[clicked_mask]\n",
    "    \n",
    "    print(f\"Training CVR model on {X_train_cvr.shape[0]} samples (clicked only)\")\n",
    "    if X_train_cvr.shape[0] > 0:\n",
    "        cvr_model.partial_fit(X_train_cvr, y_train_cvr, classes=[0, 1])\n",
    "    \n",
    "    # Evaluate on the full batch (just for logging, optional)\n",
    "    if np.sum(y_clk) > 0 and np.sum(y_clk) < len(y_clk):\n",
    "        try:\n",
    "            pred = ctr_model.predict_proba(X_sparse)[:, 1]\n",
    "            print(f\"Batch CTR Log Loss (on full batch): {log_loss(y_clk, pred):.4f}\")\n",
    "        except:\n",
    "            pass\n",
    "\n",
    "print(\"Training models...\")\n",
    "train_models(df)\n",
    "print(\"Training complete.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Export Weights\n",
    "\n",
    "We need to export the learned weights to CSV files so the Java inference engine can load them.\n",
    "The format will be:\n",
    "`weight` (one per line, corresponding to index 0 to HASH_SIZE-1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def export_weights(model, filename):\n",
    "    # Ensure we have weights (if model wasn't trained due to empty data, init zero weights)\n",
    "    if not hasattr(model, \"coef_\"):\n",
    "        model.partial_fit(csr_matrix((1, HASH_SIZE)), [0], classes=[0, 1])\n",
    "        \n",
    "    weights = model.coef_[0]\n",
    "    \n",
    "    # Create a DataFrame for fast export\n",
    "    df_weights = pl.DataFrame({\n",
    "        \"weight\": weights\n",
    "    })\n",
    "    \n",
    "    # Export without header, just the values\n",
    "    df_weights.write_csv(filename, include_header=False)\n",
    "    print(f\"Exported {filename} with shape {df_weights.shape}\")\n",
    "\n",
    "export_weights(ctr_model, \"ctr_weights.csv\")\n",
    "export_weights(cvr_model, \"cvr_weights.csv\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open('d:/4sem/project/python_training/train_models.ipynb', 'w') as f:
    json.dump(notebook_content, f, indent=2)
