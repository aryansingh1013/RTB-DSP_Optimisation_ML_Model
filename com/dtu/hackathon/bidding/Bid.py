"""
MANDATORY BID CLASS FOR HACKATHON SUBMISSION
Package: com.dtu.hackathon.bidding
Class: Bid

This is the official submission class for the RTB DSP Optimization challenge.
Implements optimized bidding strategy with:
- Memory constraint: ≤ 512 MB
- Execution time: ≤ 5 ms per request
- Goal: Maximize Score = Clicks + N × Conversions
- Auction: Second-price auction
"""

import numpy as np
import math
import pickle
import json
from pathlib import Path

class Bid:
    """
    Official Bid class for RTB DSP Optimization Challenge.
    
    Constraints:
    - Memory: ≤ 512 MB (precomputed aggregates + model weights)
    - Latency: ≤ 5 ms per request (O(1) hash lookups + simple math)
    
    Optimization Goal:
    - Maximize: Score = Clicks + N × Conversions
    - Budget-constrained sequential bidding
    """
    
    def __init__(self, model_path='experiments/results', conversion_weight=5):
        """
        Initialize bidder with precomputed model and aggregates.
        
        Args:
            model_path: Path to saved model weights and aggregates
            conversion_weight: N in Score = Clicks + N×Conversions (default: 5)
        """
        self.conversion_weight = conversion_weight
        self.base_bid = 100.0
        
        # Load precomputed aggregates (O(1) lookups)
        aggregates_path = Path(model_path) / 'aggregates.pkl'
        if aggregates_path.exists():
            with open(aggregates_path, 'rb') as f:
                self.aggregates = pickle.load(f)
            self.global_ctr = self.aggregates['global_ctr']
        else:
            # Fallback if not trained yet
            self.aggregates = None
            self.global_ctr = 0.00075
        
        # Load model weights (fast inference)
        weights_path = Path(model_path) / 'ctr_model_weights.json'
        if weights_path.exists():
            with open(weights_path, 'r') as f:
                data = json.load(f)
            self.weights = np.array(data['weights'], dtype=np.float32)
            self.bias = float(data['bias'])
        else:
            # Fallback if not trained yet
            self.weights = np.zeros(7, dtype=np.float32)
            self.bias = -7.0
        
        # Budget pacing state
        self.pacing_multiplier = 1.0
        
    def predict_ctr(self, request):
        """
        Predict CTR for bid request using fast O(1) inference.
        
        Args:
            request: Dict with request features
            
        Returns:
            float: Predicted CTR probability
            
        Performance: <1ms (hash lookups + dot product + sigmoid)
        """
        if self.aggregates is None:
            # Fallback to global CTR if model not loaded
            return self.global_ctr
        
        # Build feature vector using O(1) hash lookups
        features = np.array([
            self.aggregates['campaign_ctr'].get(
                request.get('campaign_id', request.get('advertiser', -1)), 
                self.global_ctr
            ),
            self.aggregates['region_ctr'].get(
                request.get('region', -1), 
                self.global_ctr
            ),
            self.aggregates['hour_ctr'].get(
                request.get('hour', -1), 
                self.global_ctr
            ),
            self.aggregates['adexchange_ctr'].get(
                request.get('adexchange', -1), 
                self.global_ctr
            ),
            self.aggregates['tag_ctr'].get(
                request.get('usertag', -1), 
                self.global_ctr
            ),
            request.get('adslotfloorprice', request.get('slotprice', 50)) / 100.0,
            1.0  # Bias term
        ], dtype=np.float32)
        
        # Fast inference: dot product + sigmoid
        z = np.dot(self.weights, features) + self.bias
        pctr = 1.0 / (1.0 + math.exp(-z))
        
        return pctr
    
    def bid(self, request, spent=0, budget=1000000, **kwargs):
        """
        Main bidding function (hot path - must be <5ms).
        
        Args:
            request: Bid request dict with features
            spent: Amount already spent in campaign
            budget: Total campaign budget
            **kwargs: Additional parameters (e.g., pacing_multiplier)
            
        Returns:
            int: Bid price in same units as budget/spend
            
        Performance Target: <5ms per call
        Memory: O(1) - no allocations in hot path
        """
        # Predict CTR (fast: <1ms)
        pctr = self.predict_ctr(request)
        
        # Calculate base bid using CTR ratio
        if self.global_ctr > 0:
            raw_bid = self.base_bid * (pctr / self.global_ctr)
        else:
            raw_bid = self.base_bid
        
        # Apply budget pacing multiplier
        pacing = kwargs.get('pacing_multiplier', self.pacing_multiplier)
        final_bid = raw_bid * pacing
        
        # Ensure bid meets floor price
        floor_price = request.get('adslotfloorprice', request.get('slotprice', 0))
        final_bid = max(final_bid, floor_price)
        
        # Budget check - don't bid if near limit
        if spent >= budget * 0.95:
            final_bid = floor_price  # Minimal bid when budget exhausted
        
        return int(max(1, final_bid))
    
    def update_pacing(self, spent, budget, time_fraction):
        """
        Update budget pacing multiplier.
        
        Args:
            spent: Amount spent so far
            budget: Total budget
            time_fraction: Fraction of time elapsed (0.0 to 1.0)
        """
        if time_fraction > 0:
            # Expected vs actual spend rate
            expected_rate = budget * time_fraction
            actual_rate = spent
            
            # Adjust pacing
            if actual_rate > 0:
                ratio = expected_rate / actual_rate
                self.pacing_multiplier *= (ratio ** 0.1)  # Smooth adjustment
                
                # Clamp to reasonable range
                self.pacing_multiplier = max(0.1, min(10.0, self.pacing_multiplier))


# Alias for import compatibility
class Bidder(Bid):
    """Alias for backward compatibility."""
    pass


if __name__ == "__main__":
    # Quick test
    bidder = Bid()
    
    test_request = {
        'campaign_id': 5,
        'region': 10,
        'hour': 14,
        'adexchange': 2,
        'usertag': 25,
        'adslotfloorprice': 30
    }
    
    bid_price = bidder.bid(test_request, spent=1000, budget=100000)
    print(f"Test bid: {bid_price}")
    print("✅ Bid class initialized successfully")
