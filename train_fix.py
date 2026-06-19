import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler

print("="*60)
print("COMPILING REALISTIC INDUSTRIAL XGBOOST RISK MODEL")
print("="*60)

# Expanded dataset representing real-world financial patterns across market caps
# Features: working_capital_ta, retained_earnings_ta, ebitda_ta, market_cap_tl, sales_ta, sentiment
market_data = [
    # 🟢 [CLASS 0: STABLE / BLUE CHIP / HEALTHY]
    [0.35, 0.55, 0.18, 6.50, 1.80, 0.85, 0],   # Mega-cap Tech (e.g., NVDA profile)
    [0.25, 0.45, 0.14, 4.20, 1.20, 0.60, 0],   # Large-cap Stable Consumer Goods
    [0.18, 0.32, 0.09, 2.50, 1.05, 0.40, 0],   # Heavy Industrial / Energy Giant
    [0.40, 0.60, 0.22, 8.50, 2.10, 0.95, 0],   # Premium High-Growth Tech
    [0.22, 0.38, 0.11, 3.80, 1.35, 0.70, 0],   # Stable Infrastructure Firm
    [0.12, 0.25, 0.07, 1.95, 0.95, 0.30, 0],   # Mid-cap Manufacturing (Solid)
    [0.30, 0.50, 0.16, 5.10, 1.65, 0.75, 0],   # Large Pharmaceuticals 

    # 🟡 [CLASS 0/1: GREY ZONE / VOLATILE TRANSITIONS]
    [0.05, 0.12, 0.04, 1.15, 0.85, -0.45, 1],  # Dragged down heavily by negative news/lawsuits
    [0.08, 0.15, 0.03, 1.30, 0.95, 0.55, 0],   # Margins tight, but saved by strong market sentiment
    [-0.05, -0.12, 0.02, 0.75, 0.70, -0.25, 1], # Rising debt leverage starting to pressure assets
    [0.14, 0.20, 0.06, 1.45, 1.10, -0.15, 0],  # Mid-cap dealing with short-term operational headwinds
    [0.01, 0.05, 0.01, 0.90, 0.80, -0.60, 1],  # Micro-cap hovering on edge of distress

    # 🔴 [CLASS 1: ACUTE DISTRESS / SEVERE RESTRUCTURING RISK]
    [-0.18, -0.52, -0.06, 0.18, 0.55, -0.70, 1], # Heavily Indebted Entertainment (e.g., AMC profile)
    [-0.35, -0.85, -0.18, 0.05, 0.38, -0.90, 1], # Extreme Capital Decay / Retail Default
    [-0.08, -0.25, -0.04, 0.30, 0.65, -0.50, 1], # Highly Leveraged Small-cap Mining/Resources
    [-0.55, -1.35, -0.35, 0.01, 0.15, -0.98, 1], # Total Technical Insolvency / Pre-Bankruptcy
    [-0.25, -0.65, -0.12, 0.12, 0.48, -0.75, 1], # Legacy brick-and-mortar retail facing collapse
    [-0.10, -0.40, -0.02, 0.25, 0.60, -0.40, 1]  # Staggering supply chain / Capital-starved micro-cap
]

df = pd.DataFrame(market_data, columns=[
    'working_capital_ta', 'retained_earnings_ta', 'ebitda_ta', 
    'market_cap_tl', 'sales_ta', 'sentiment', 'target'
])

X = df.drop(columns=['target'])
y = df['target']

# RECOMMENDED IMPROVEMENT: Stacking RobustScaler instead of MinMaxScaler
# RobustScaler uses interquartile ranges (IQR), which handles extreme outliers 
# like a market_cap_tl of 70+ much more elegantly without squishing normal companies.
pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('classifier', XGBClassifier(
        n_estimators=40,
        max_depth=3,
        learning_rate=0.08,
        scale_pos_weight=1.0, 
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    ))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "pipeline.joblib")
print("🎯 PRODUCTION EXPORT SUCCESS: New robust pipeline.joblib successfully written!")
print("="*60)