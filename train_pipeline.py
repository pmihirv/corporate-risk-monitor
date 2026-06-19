import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

print("Initializing upgraded synthetic training dataset generation...")
np.random.seed(42)
num_samples = 2000

is_distressed = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])

working_capital_ta = np.where(is_distressed == 1, np.random.uniform(-0.2, 0.1, num_samples), np.random.uniform(0.1, 0.5, num_samples))
retained_earnings_ta = np.where(is_distressed == 1, np.random.uniform(-0.5, 0.0, num_samples), np.random.uniform(0.1, 0.6, num_samples))
ebitda_ta = np.where(is_distressed == 1, np.random.uniform(-0.3, 0.05, num_samples), np.random.uniform(0.05, 0.4, num_samples))
market_cap_tl = np.where(is_distressed == 1, np.random.uniform(0.1, 0.5, num_samples), np.random.uniform(0.6, 3.0, num_samples))
sales_ta = np.where(is_distressed == 1, np.random.uniform(0.2, 1.0, num_samples), np.random.uniform(0.8, 2.5, num_samples))
sentiment = np.where(is_distressed == 1, np.random.uniform(-0.8, 0.1, num_samples), np.random.uniform(-0.1, 0.9, num_samples))

df = pd.DataFrame({
    'working_capital_ta': working_capital_ta,
    'retained_earnings_ta': retained_earnings_ta,
    'ebitda_ta': ebitda_ta,
    'market_cap_tl': market_cap_tl,
    'sales_ta': sales_ta,
    'sentiment': sentiment,
    'risk_label': is_distressed
})

X = df.drop(columns=['risk_label'])
y = df['risk_label']

imbalance_ratio = np.sum(y == 0) / np.sum(y == 1)

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', XGBClassifier(
        scale_pos_weight=imbalance_ratio,
        max_depth=4,
        learning_rate=0.05,
        n_estimators=150,
        eval_metric='logloss',
        random_state=42
    ))
])

pipeline.fit(X, y)
joblib.dump(pipeline, 'pipeline.joblib')
print("SUCCESS: Full 5-Ratio + Sentiment pipeline compiled into 'pipeline.joblib'!")