import joblib
import os

try:
    config = joblib.load("d:/4sem/project/feature_config.pkl")
    print("Feature Config Keys:", config.keys())
    print("Metrics:", config.get('metrics', 'No metrics key found'))
except Exception as e:
    print(f"Error loading config: {e}")
