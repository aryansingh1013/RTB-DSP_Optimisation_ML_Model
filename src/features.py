"""Feature engineering for RTB DSP system."""
import pandas as pd
import numpy as np
import pickle
import os
from src.config import PRIOR_ALPHA

class FeatureEngineer:
    def __init__(self, prior_alpha=PRIOR_ALPHA):
        self.prior_alpha = prior_alpha
        self.aggregates = {}
        
    def build_aggregates(self, df):
        print("Building feature aggregates...")
        global_ctr = df['click'].mean()
        self.aggregates['global_ctr'] = global_ctr
        
        for group_col, agg_name in [('campaign_id', 'campaign_ctr'), ('region', 'region_ctr'), 
                                     ('hour', 'hour_ctr'), ('adexchange', 'adexchange_ctr'), ('usertag', 'tag_ctr')]:
            stats = df.groupby(group_col).agg({'click': ['sum', 'count']}).reset_index()
            stats.columns = [group_col, 'clicks', 'count']
            stats['ctr'] = self._smooth_ctr(stats['clicks'], stats['count'], global_ctr)
            self.aggregates[agg_name] = dict(zip(stats[group_col], stats['ctr']))
        
        print(f"Built aggregates: {list(self.aggregates.keys())}")
        return self.aggregates
    
    def _smooth_ctr(self, clicks, counts, prior_mean):
        raw_ctr = clicks / counts
        return (counts * raw_ctr + self.prior_alpha * prior_mean) / (counts + self.prior_alpha)
    
    def create_training_features(self, df):
        """Create features for training - MUST match Bid class feature vector (7 features)."""
        features = pd.DataFrame()
        
        # IMPORTANT: Order and count must match Bid.predict_ctr() feature vector
        # Feature engineering creates 7 features to match runtime inference:
        # 1. campaign_ctr
        # 2. region_ctr  
        # 3. hour_ctr
        # 4. adexchange_ctr
        # 5. tag_ctr
        # 6. floor_price (normalized)
        # 7. bias (constant 1.0)
        
        if self.aggregates:
            features['campaign_ctr'] = df['campaign_id'].map(self.aggregates.get('campaign_ctr', {})).fillna(self.aggregates['global_ctr'])
            features['region_ctr'] = df['region'].map(self.aggregates.get('region_ctr', {})).fillna(self.aggregates['global_ctr'])
            features['hour_ctr'] = df['hour'].map(self.aggregates.get('hour_ctr', {})).fillna(self.aggregates['global_ctr'])
            features['adexchange_ctr'] = df['adexchange'].map(self.aggregates.get('adexchange_ctr', {})).fillna(self.aggregates['global_ctr'])
            features['tag_ctr'] = df['usertag'].map(self.aggregates.get('tag_ctr', {})).fillna(self.aggregates['global_ctr'])
        else:
            # Fallback if aggregates not computed
            global_ctr = 0.00075
            features['campaign_ctr'] = global_ctr
            features['region_ctr'] = global_ctr
            features['hour_ctr'] = global_ctr
            features['adexchange_ctr'] = global_ctr
            features['tag_ctr'] = global_ctr
        
        features['floor_price'] = df['adslotfloorprice'] / 100.0
        features['bias'] = 1.0
        
        return features
    
    def create_runtime_features(self, request):
        global_ctr = self.aggregates['global_ctr']
        return np.array([
            self.aggregates['campaign_ctr'].get(request.get('campaign_id', -1), global_ctr),
            self.aggregates['region_ctr'].get(request.get('region', -1), global_ctr),
            self.aggregates['hour_ctr'].get(request.get('hour', -1), global_ctr),
            self.aggregates['adexchange_ctr'].get(request.get('adexchange', -1), global_ctr),
            self.aggregates['tag_ctr'].get(request.get('usertag', -1), global_ctr),
            request.get('adslotfloorprice', 50) / 100.0,
            1.0
        ], dtype=np.float32)
    
    def save_aggregates(self, path='experiments/results/aggregates.pkl'):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.aggregates, f)
        print(f"Saved aggregates to {path}")
    
    def load_aggregates(self, path='experiments/results/aggregates.pkl'):
        with open(path, 'rb') as f:
            self.aggregates = pickle.load(f)
        print(f"Loaded aggregates from {path}")
        return self.aggregates
