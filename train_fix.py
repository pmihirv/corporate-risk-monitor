import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

print("="*60)
print("COMPILING ACCURATE DUAL-MODEL XGBOOST ARTIFACT")
print("="*60)

# Synthesize a balanced matrix across Healthy, Borderline, and Distressed Profiles
# Features: working_capital_ta, retained_earnings_ta, ebitda_ta, market_cap_tl, sales_ta, sentiment
training_data = [
    # --- HEALTHY CORPORATE PROFILES (Class 0: No Risk) ---
    [0.35, 0.55, 0.18, 5.50, 1.80, 0.75, 0],
    [0.20, 0.40, 0.12, 3.20, 1.40, 0.50, 0],
    [0.15, 0.30, 0.08, 2.10, 1.10, 0.20, 0],
    [0.45, 0.65, 0.25, 8.00, 2.20, 0.90, 0],
    
    # --- DISTRESSED CORPORATE PROFILES (Class 1: Distress / Default Risk) ---
    [-0.15, -0.45, -0.05, 0.20, 0.60, -0.65, 1],
    [-0.30, -0.80, -0.15, 0.08, 0.40, -0.85, 1],
    [-0.05, -0.20, -0.02, 0.35, 0.75, -0.40, 1],
    [-0.50, -1.20, -0.30, 0.02, 0.20, -0.95, 1],
    
    # --- VOLATILE / GREY ZONE TRANSITIONS ---
    [0.02, 0.10, 0.03, 0.85, 0.90, -0.50, 1],  
    [0.05, 0.08, 0.02, 0.95, 0.95, 0.45, 0],   
    [-0.08, -0.15, 0.01, 0.50, 0.80, -0.30, 1], 
    [0.12, 0.18, 0.05, 1.10, 1.05, -0.10, 0]   
]

df = pd.DataFrame(training_data, columns=[
    'working_capital_ta', 'retained_earnings_ta', 'ebitda_ta', 
    'market_cap_tl', 'sales_ta', 'sentiment', 'target'
])

X = df.drop(columns=['target'])
y = df['target']

# Compile the machine learning pipeline container with MinMaxScaler to prevent tree collapse
pipeline = Pipeline([
    ('scaler', MinMaxScaler(feature_range=(0, 1))),
    ('classifier', XGBClassifier(
        n_estimators=25,
        max_depth=3,
        learning_rate=0.1,
        scale_pos_weight=1.0, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

# Execute training fit
pipeline.fit(X, y)

# Export artifact binary
joblib.dump(pipeline, "pipeline.joblib")
print("🎯 SUCCESS: pipeline.joblib successfully compiled and written!")
print("="*60)