# Configuration for RTB DSP Optimization

# Base bidding parameters
BASE_BID = 100.0
AVG_CTR = 0.00075
CVR_WEIGHT = 5.0  # Weight for conversions in KPI

# Budget proportions for testing
BUDGET_PROPORTIONS = [1/32, 1/8, 1/2]

# Pacing parameters
PACING_UPDATE_WINDOW = 1000
PACING_BETA = 0.1
PACING_MIN = 0.1
PACING_MAX = 10.0

# Feature engineering
PRIOR_ALPHA = 10

# Model parameters
LR_C = 1.0
LR_MAX_ITER = 200

# Dataset simulation parameters
N_SAMPLES = 100000
N_CAMPAIGNS = 20
N_REGIONS = 50
N_HOURS = 24
GLOBAL_CTR_MEAN = 0.00075
GLOBAL_CVR_MEAN = 0.0001
