import polars as pl
import numpy as np
import joblib
import os
from sklearn.feature_extraction import FeatureHasher
from datetime import datetime

# HASH_SIZE is now determined by the hasher object loaded from disk
# default fallback if not loaded
HASH_SIZE = 2**20

class InferenceEngine:
    def __init__(self, ctr_model_path, cvr_model_path, hasher_path=None, hasher_cvr_path=None, config_path=None, config_cvr_path=None):
        self.ctr_model = None
        self.cvr_model = None
        self.hasher = None
        self.hasher_cvr = None
        
        # Load CTR Model
        if ctr_model_path and os.path.exists(ctr_model_path):
            try:
                self.ctr_model = joblib.load(ctr_model_path)
            except Exception as e:
                print(f"Error loading CTR model: {e}")

        if hasher_path and os.path.exists(hasher_path):
            try:
                self.hasher = joblib.load(hasher_path)
            except Exception as e:
                print(f"Error loading Hasher: {e}")

        # Load CVR Model (New)
        if cvr_model_path and os.path.exists(cvr_model_path):
            try:
                self.cvr_model = joblib.load(cvr_model_path)
                print(f"Loaded CVR model from {cvr_model_path}")
            except Exception as e:
                print(f"Error loading CVR model: {e}")
                # Fallback to old weights if path was actually a CSV
                if cvr_model_path.endswith('.csv'):
                     self.cvr_weights = self._load_linear_weights(cvr_model_path)

        if hasher_cvr_path and os.path.exists(hasher_cvr_path):
            try:
                self.hasher_cvr = joblib.load(hasher_cvr_path)
                print(f"Loaded CVR Hasher from {hasher_cvr_path}")
            except Exception as e:
                print(f"Error loading CVR Hasher: {e}")
                
        print("Models loaded.")

    def _load_linear_weights(self, path):
        try:
            if os.path.exists(path):
                df = pl.read_csv(path, has_header=False)
                return df["column_1"].to_numpy()
        except Exception as e:
            print(f"Error loading {path}, using random weights: {e}")
        return np.random.randn(HASH_SIZE) * 0.001

    def get_features_dict_ctr(self, row):
        # CTR Model uses generic single features 1..20
        features = {}
        for i in range(1, 21):
            col_name = str(i)
            if col_name in row and row[col_name] is not None:
                val = str(row[col_name])
                if val.lower() not in ['none', 'nan', 'null', '']:
                     features[f"{col_name}={val}"] = 1.0
        return features

    def get_features_dict_cvr(self, row):
        # CVR Model uses Feature Engineering:
        # - Hour, Weekday from '1'
        # - Cross Features: 6_7, 19_6, 4_Hour, 14_16
        
        # Helper to safely get string value
        def get_val(idx):
            val = row.get(str(idx))
            if val is None: return "0"
            return str(val)

        # 1. Base Features (excluding some IDs as per training script)
        # In script we excluded: [1, 3, 5, 9, 10, 17, 18]
        # Keep: 2, 4, 6, 7, 8, 11, 12, 13, 14, 15, 16, 19, 20
        features = {}
        exclude = [1, 3, 5, 9, 10, 17, 18]
        for i in range(1, 21):
            if i in exclude: continue
            col_name = str(i)
            val = get_val(i)
            if val.lower() not in ['none', 'nan', 'null', '', '0']:
                features[f"{col_name}={val}"] = 1.0

        # 2. Extract Time Features
        ts = get_val(1)
        hour = "0"
        # weekday = "0" # unused in training script logic (only Hour extracted effectively)
        if len(ts) >= 10:
            try:
                # YYYYMMDDHH...
                dt = datetime(int(ts[0:4]), int(ts[4:6]), int(ts[6:8]), int(ts[8:10]))
                hour = str(dt.hour)
            except: pass
        
        features[f"feature_hour={hour}"] = 1.0
        
        # 3. Cross Features
        # Region(6) x City(7)
        r = get_val(6)
        c = get_val(7)
        features[f"cross_region_city={r}_{c}"] = 1.0
        
        # Advertiser(19) x Region(6)
        adv = get_val(19)
        features[f"cross_adv_region={adv}_{r}"] = 1.0
        
        # UserAgent(4) x Hour
        ua = get_val(4)
        features[f"cross_ua_hour={ua}_{hour}"] = 1.0
        
        # SlotFormat(14) x Creative(16)
        fmt = get_val(14)
        crt = get_val(16)
        features[f"cross_format_creative={fmt}_{crt}"] = 1.0
        
        return features

    def predict(self, row):
        # 1. CTR Prediction
        p_ctr = 0.001
        if self.ctr_model and self.hasher:
            try:
                feats = self.get_features_dict_ctr(row)
                X = self.hasher.transform([feats])
                p_ctr = self.ctr_model.predict_proba(X)[0, 1]
            except: pass
        
        # 2. CVR Prediction
        p_cvr = 0.0
        if self.cvr_model and self.hasher_cvr:
            try:
                feats = self.get_features_dict_cvr(row)
                X = self.hasher_cvr.transform([feats])
                p_cvr = self.cvr_model.predict_proba(X)[0, 1]
            except Exception as e:
                # print(f"CVR Error: {e}")
                pass
        elif hasattr(self, 'cvr_weights'): 
            # Fallback to old linear model if new one not loaded
            try:
                 # reusing simple hash for fallback
                indices = []
                import mmh3
                for i in range(1, 21):
                    val = str(row.get(str(i), '0'))
                    idx = mmh3.hash(f"{i}={val}", seed=42, signed=False) % HASH_SIZE
                    indices.append(idx)
                if len(indices) > 0:
                    logit = np.sum(self.cvr_weights[indices])
                    p_cvr = 1 / (1 + np.exp(-logit))
            except: pass
        
        # Calculate Unconditional CVR (P(Conversion) = P(Click) * P(Conversion|Click))
        p_cvr_unconditional = p_ctr * p_cvr
        
        return p_ctr, p_cvr, p_cvr_unconditional

    def bid(self, p_ctr, base_bid, avg_ctr=0.001):
        if avg_ctr == 0: avg_ctr = 0.001
        return int(base_bid * (p_ctr / avg_ctr))

class PacingController:
    """
    PID Pacing Controller ported from Bid.java.
    Ensures budget lasts the entire simulation by adjusting bid prices.
    """
    def __init__(self, total_budget, total_steps):
        self.total_budget = total_budget
        self.total_steps = total_steps
        self.actual_spend = 0.0
        
        # PID State
        self.pacing_factor = 1.0
        self.integral_error = 0.0
        self.previous_error = 0.0
        
        # PID Gains (from Bid.java)
        self.Kp = 0.05
        self.Ki = 0.001
        self.Kd = 0.01
        
    def update(self, spent_in_step, current_step_index):
        """
        Update state using PID logic.
        Matches updatePacingMultiplier() in Bid.java but uses steps instead of time.
        """
        self.actual_spend += spent_in_step
        
        # Avoid division by zero
        if current_step_index <= 0:
            return 1.0
            
        # Calculate target spend at current step
        # Target = (TotalBudget / TotalSteps) * CurrentStep
        target_spend = (self.total_budget / self.total_steps) * current_step_index
        
        # Error: How much we are UNDER-spending 
        # (Positive error means we have budget left -> increase bid)
        error = target_spend - self.actual_spend
        
        # PID Update
        self.integral_error += error
        derivative = error - self.previous_error
        
        # Discrete PID Equation
        # pacingMultiplier = 1.0 + (Kp * error) + (Ki * integralError) + (Kd * derivative)
        # Note: In Java code, error is (target - actual). 
        # If target (100) > actual (50), error is 50. Multiplier increases. Correct.
        
        # We need to normalize error relative to budget tick size to keep Kp generic?
        # Java code didn't normalize. Let's stick to raw values but watch out for scale.
        # If budget is 10,000, error could be 1,000. 
        # Kp * error = 0.05 * 1000 = 50. Multiplier becomes 51.0! 
        # That seems aggressive. 
        # BUT, if `Bid.java` worked, maybe input was scaled? 
        # Or maybe the loop was very fast (ms) so error was small per tick.
        # Here we step row by row. 
        # Let's normalize error by (TotalBudget/TotalSteps) i.e. avg_spend_per_step?
        # Or just clamp aggressively. Java clamped [0.001, 100.0].
        
        # Let's try raw first as requested "Port exact logic".
        # But I suspect we might need to scale Kp if step size differs (ms vs row).
        # Let's scale error down by avg_bid_price (~100) or just trust the clamp.
        
        # Actually, let's normalize error relative to the "expected spend per step".
        # error_norm = error / (self.total_budget / self.total_steps)
        # No, strict port first.
        
        raw_adjustment = (self.Kp * error) + (self.Ki * self.integral_error) + (self.Kd * derivative)
        
        # In our case, error might be large (e.g. 100 CPM). 
        # 0.05 * 100 = 5. Factor becomes 6. That's fine.
        
        self.pacing_factor = 1.0 + raw_adjustment
        
        # Safety bounds (from Bid.java)
        self.pacing_factor = max(0.001, min(self.pacing_factor, 100.0))
        
        self.previous_error = error
        return self.pacing_factor
