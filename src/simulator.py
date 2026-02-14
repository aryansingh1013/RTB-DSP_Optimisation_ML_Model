"""Auction simulator for RTB DSP system."""
import pandas as pd
import numpy as np
from tqdm import tqdm
from src.config import CVR_WEIGHT

class AuctionSimulator:
    def __init__(self, bidder, budget, strategy_name="Unknown"):
        self.bidder, self.budget, self.strategy_name = bidder, budget, strategy_name
        self.reset_metrics()
    
    def reset_metrics(self):
        self.spent = self.impressions = self.clicks = self.conversions = self.wins = self.total_requests = 0
        self.history = []
    
    def simulate(self, df, verbose=True):
        self.reset_metrics()
        df = df.sort_values('timestamp').reset_index(drop=True)
        iterator = tqdm(df.iterrows(), total=len(df), desc=f"Simulating {self.strategy_name}") if verbose else df.iterrows()
        
        for idx, request in iterator:
            self.total_requests += 1
            if self.spent >= self.budget:
                break
            
            req_dict = request.to_dict()
            # Handle both Bid class (has predict_ctr) and raw strategy classes
            if hasattr(self.bidder, 'predict_ctr'):
                # This is a Bid class instance
                bid_price = self.bidder.bid(req_dict, self.spent, self.budget)
            elif hasattr(self.bidder, 'bid'):
                # This is a strategy class instance
                bid_price = self.bidder.bid(req_dict, 0.0, self.spent, self.budget)
            else:
                # This is a callable
                bid_price = self.bidder(req_dict, self.spent, self.budget)
            
            if bid_price <= request['adslotfloorprice']:
                continue
            
            if bid_price > request['payingprice']:
                self.wins += 1
                self.spent += request['payingprice']
                self.impressions += 1
                if request['click'] == 1:
                    self.clicks += 1
                if request['conversion'] == 1:
                    self.conversions += 1
                self.history.append({
                    'timestamp': request['timestamp'], 'bid': bid_price, 'price_paid': request['payingprice'],
                    'click': request['click'], 'conversion': request['conversion']
                })
        
        kpi = self.clicks + CVR_WEIGHT * self.conversions
        return {
            'strategy': self.strategy_name, 'budget': self.budget, 'spent': self.spent,
            'impressions': self.impressions, 'clicks': self.clicks, 'conversions': self.conversions,
            'kpi': kpi, 'win_rate': self.wins / max(1, self.total_requests),
            'ctr': self.clicks / max(1, self.impressions), 'cvr': self.conversions / max(1, self.clicks),
            'cpc': self.spent / max(1, self.clicks), 'cpa': self.spent / max(1, self.conversions) if self.conversions > 0 else 0
        }
    
    def get_history_df(self):
        return pd.DataFrame(self.history)

def run_multi_strategy_simulation(df, strategies, budgets, verbose=True):
    results = []
    for budget in budgets:
        for strategy_name, bidder in strategies.items():
            sim = AuctionSimulator(bidder, budget, strategy_name)
            results.append(sim.simulate(df, verbose=verbose))
    return pd.DataFrame(results)
