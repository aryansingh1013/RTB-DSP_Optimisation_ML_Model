"""Bidding strategies for RTB DSP system."""
import numpy as np
import math
from src.config import BASE_BID, AVG_CTR

class BiddingStrategy:
    def __init__(self, name):
        self.name = name
    def bid(self, request, pctr, spent, budget, **kwargs):
        raise NotImplementedError

class ConstantBidding(BiddingStrategy):
    def __init__(self, bid_price=50):
        super().__init__("Constant")
        self.bid_price = bid_price
    def bid(self, request, pctr, spent, budget, **kwargs):
        return self.bid_price

class RandomBidding(BiddingStrategy):
    def __init__(self, min_bid=10, max_bid=200):
        super().__init__("Random")
        self.min_bid, self.max_bid = min_bid, max_bid
    def bid(self, request, pctr, spent, budget, **kwargs):
        return int(np.random.uniform(self.min_bid, self.max_bid))

class MCPCBidding(BiddingStrategy):
    def __init__(self, max_ecpc=1000):
        super().__init__("MCPC")
        self.max_ecpc = max_ecpc
    def bid(self, request, pctr, spent, budget, **kwargs):
        return int(self.max_ecpc * pctr)

class LinearBidding(BiddingStrategy):
    def __init__(self, base_bid=BASE_BID, avg_ctr=AVG_CTR):
        super().__init__("Linear")
        self.base_bid, self.avg_ctr = base_bid, avg_ctr
    def bid(self, request, pctr, spent, budget, **kwargs):
        if self.avg_ctr == 0:
            return int(self.base_bid)
        return int(self.base_bid * (pctr / self.avg_ctr))

class MLLinearBidding(BiddingStrategy):
    def __init__(self, base_bid=BASE_BID, avg_ctr=AVG_CTR, pacing_enabled=True):
        super().__init__("ML-Linear")
        self.base_bid, self.avg_ctr, self.pacing_enabled = base_bid, avg_ctr, pacing_enabled
        self.pacing_multiplier = 1.0
    def bid(self, request, pctr, spent, budget, **kwargs):
        raw_bid = self.base_bid if self.avg_ctr == 0 else self.base_bid * (pctr / self.avg_ctr)
        if self.pacing_enabled:
            pacing_multiplier = kwargs.get('pacing_multiplier', self.pacing_multiplier)
            final_bid = raw_bid * pacing_multiplier
        else:
            final_bid = raw_bid
        return int(max(1, final_bid))

class Bid:
    def __init__(self, aggregates, weights, bias, strategy='ml-linear'):
        self.aggregates, self.weights, self.bias = aggregates, weights, bias
        self.global_ctr = aggregates['global_ctr']
        if strategy == 'constant':
            self.strategy = ConstantBidding()
        elif strategy == 'random':
            self.strategy = RandomBidding()
        elif strategy == 'mcpc':
            self.strategy = MCPCBidding()
        elif strategy == 'linear':
            self.strategy = LinearBidding()
        else:
            self.strategy = MLLinearBidding()
    
    def predict_ctr(self, request):
        features = [
            self.aggregates['campaign_ctr'].get(request.get('campaign_id', -1), self.global_ctr),
            self.aggregates['region_ctr'].get(request.get('region', -1), self.global_ctr),
            self.aggregates['hour_ctr'].get(request.get('hour', -1), self.global_ctr),
            self.aggregates['adexchange_ctr'].get(request.get('adexchange', -1), self.global_ctr),
            self.aggregates['tag_ctr'].get(request.get('usertag', -1), self.global_ctr),
            request.get('adslotfloorprice', 50) / 100.0,
            1.0
        ]
        z = sum(w * f for w, f in zip(self.weights, features)) + self.bias
        return 1.0 / (1.0 + math.exp(-z))
    
    def bid(self, request, spent=0, budget=1000000, **kwargs):
        pctr = self.predict_ctr(request)
        bid_price = self.strategy.bid(request, pctr, spent, budget, **kwargs)
        floor_price = request.get('adslotfloorprice', 0)
        return int(max(bid_price, floor_price))
