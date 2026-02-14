import sys
import os
import polars as pl

# Add dashboard dir to path to import utils
sys.path.append(os.path.join(os.getcwd(), 'dashboard'))
from utils import InferenceEngine

print("Initializing InferenceEngine...")
try:
    # mocking paths relative to where we run this script (project root)
    engine = InferenceEngine(
        ctr_model_path="ctr_model_calibrated.pkl",
        cvr_model_path="cvr_model_calibrated.pkl",
        hasher_path="feature_hasher.pkl",
        hasher_cvr_path="feature_hasher_cvr.pkl",
        config_path="feature_config.pkl",
        config_cvr_path="feature_config_cvr.pkl"
    )
    
    # Create a dummy row used for inference
    # Must include '1' for hour extraction
    row = {
        '1': '20130606000133', 
        '6': '1', 
        '7': '1', 
        '19': 'abc', 
        '4': 'mozilla', 
        '14': '1', 
        '16': 'creative_1'
    }
    
    print("Testing prediction on dummy row...")
    p_ctr, p_cvr = engine.predict(row)
    print(f"pCTR: {p_ctr:.6f}")
    print(f"pCVR: {p_cvr:.6f}")
    
    if p_cvr > 0 and p_cvr < 1:
        print("Verification Successful: CVR prediction is a valid probability.")
    else:
        print(f"Verification Warning: CVR prediction {p_cvr} seems odd.")

except Exception as e:
    print(f"Verification Failed: {e}")
    import traceback
    traceback.print_exc()
