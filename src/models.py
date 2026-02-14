"""CTR prediction models for RTB DSP system."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, log_loss
import joblib
import os
import json
import math

class CTRModel:
    def __init__(self, C=1.0, max_iter=200):
        self.model = LogisticRegression(C=C, max_iter=max_iter, solver='saga', random_state=42)
        self.weights, self.bias, self.feature_names = None, None, None
        
    def train(self, X, y):
        print("Training CTR model...")
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X = X.values
        self.model.fit(X, y)
        self.weights, self.bias = self.model.coef_[0], self.model.intercept_[0]
        
        train_pred = self.model.predict_proba(X)[:, 1]
        train_auc, train_logloss = roc_auc_score(y, train_pred), log_loss(y, train_pred)
        print(f"Training AUC: {train_auc:.4f}, LogLoss: {train_logloss:.4f}")
        return {'auc': train_auc, 'logloss': train_logloss}
    
    def evaluate(self, X, y):
        if isinstance(X, pd.DataFrame):
            X = X.values
        pred = self.model.predict_proba(X)[:, 1]
        auc, logloss = roc_auc_score(y, pred), log_loss(y, pred)
        print(f"Test AUC: {auc:.4f}, LogLoss: {logloss:.4f}")
        return {'auc': auc, 'logloss': logloss, 'predictions': pred}
    
    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
        return self.model.predict_proba(X)[:, 1]
    
    def predict_proba_fast(self, feature_vector):
        if self.weights is None:
            raise ValueError("Model not trained")
        z = np.dot(self.weights, feature_vector) + self.bias
        return 1.0 / (1.0 + math.exp(-z))
    
    def save(self, path='experiments/results/ctr_model.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.model, path)
        weights_path = path.replace('.pkl', '_weights.json')
        with open(weights_path, 'w') as f:
            json.dump({'weights': self.weights.tolist(), 'bias': float(self.bias), 'feature_names': self.feature_names}, f, indent=2)
        print(f"Saved model to {path} and weights to {weights_path}")
    
    def load(self, path='experiments/results/ctr_model.pkl'):
        self.model = joblib.load(path)
        self.weights, self.bias = self.model.coef_[0], self.model.intercept_[0]
        print(f"Loaded model from {path}")
    
    def load_weights(self, path='experiments/results/ctr_model_weights.json'):
        with open(path, 'r') as f:
            data = json.load(f)
        self.weights, self.bias, self.feature_names = np.array(data['weights']), data['bias'], data.get('feature_names')
        print(f"Loaded weights from {path}")
