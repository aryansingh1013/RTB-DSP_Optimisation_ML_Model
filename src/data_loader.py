"""Data generation and loading for RTB DSP system."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

class SyntheticDataGenerator:
    def __init__(self, n_samples=100000, seed=42):
        self.n_samples = n_samples
        self.seed = seed
        np.random.seed(seed)
        
    def generate(self):
        n_campaigns, n_regions, n_adexchanges, n_adslots = 20, 50, 5, 100
        start_time = datetime(2024, 1, 1)
        timestamps = [start_time + timedelta(seconds=i*10) for i in range(self.n_samples)]
        
        data = {
            'bidid': [f'bid_{i:010d}' for i in range(self.n_samples)],
            'timestamp': timestamps,
            'campaign_id': np.random.randint(0, n_campaigns, self.n_samples),
            'region': np.random.randint(0, n_regions, self.n_samples),
            'adexchange': np.random.randint(0, n_adexchanges, self.n_samples),
            'adslotid': np.random.randint(0, n_adslots, self.n_samples),
            'adslotwidth': np.random.choice([300, 728, 160, 250], self.n_samples),
            'adslotheight': np.random.choice([250, 90, 600, 250], self.n_samples),
            'adslotfloorprice': np.random.randint(10, 100, self.n_samples),
            'usertag': np.random.randint(0, 200, self.n_samples),
        }
        
        df = pd.DataFrame(data)
        df['hour'] = df['timestamp'].dt.hour
        df['weekday'] = df['timestamp'].dt.weekday
        df['day'] = df['timestamp'].dt.day
        
        campaign_ctr = np.random.beta(2, 2000, n_campaigns)
        region_ctr = np.random.beta(2, 2000, n_regions)
        hour_effect = np.sin(np.arange(24) * 2 * np.pi / 24) * 0.0003 + 0.0007
        hour_ctr = hour_effect / hour_effect.mean() * 0.00075
        
        df['expected_ctr'] = (campaign_ctr[df['campaign_id']] * 0.4 + region_ctr[df['region']] * 0.3 + hour_ctr[df['hour']] * 0.3)
        df['click'] = (np.random.random(self.n_samples) < df['expected_ctr']).astype(int)
        df['conversion'] = 0
        clicked_mask = df['click'] == 1
        df.loc[clicked_mask, 'conversion'] = (np.random.random(clicked_mask.sum()) < 0.1).astype(int)
        df['payingprice'] = (df['adslotfloorprice'] + np.random.gamma(2, 20, self.n_samples)).astype(int)
        df['biddingprice'] = (df['payingprice'] * np.random.uniform(1.0, 1.3, self.n_samples)).astype(int)
        df = df.drop('expected_ctr', axis=1)
        return df
    
    def save(self, df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)
        print(f"Saved {len(df)} records to {path}")
        
    def load(self, path):
        return pd.read_csv(path, parse_dates=['timestamp'])

def load_data(data_path='data/synthetic_rtb.csv', regenerate=False):
    if not os.path.exists(data_path) or regenerate:
        print("Generating synthetic RTB dataset...")
        generator = SyntheticDataGenerator(n_samples=100000)
        df = generator.generate()
        generator.save(df, data_path)
    else:
        print(f"Loading dataset from {data_path}...")
        generator = SyntheticDataGenerator()
        df = generator.load(data_path)
    print(f"Loaded {len(df)} bid requests - CTR: {df['click'].mean():.4%}, CVR: {df['conversion'].mean():.4%}")
    return df

if __name__ == "__main__":
    df = load_data(regenerate=True)
    print(df.head())
