import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
import joblib

print('Simulating corporate health telemetry records...')
np.random.seed(42)
n_samples = 5000
X_financials = np.random.normal(loc=[0.12, 0.25, 0.04], scale=[0.08, 0.12, 0.04], size=(n_samples, 3))
X_sentiment = np.random.uniform(low=-0.9, high=0.7, size=(n_samples, 1))
X_combined = np.hstack((X_financials, X_sentiment))

risk_formula = (X_combined[:, 0] * -3.8) + (X_combined[:, 1] * -2.2) + (X_combined[:, 2] * -4.5) + (X_combined[:, 3] * -2.8)
probabilities = 1 / (1 + np.exp(-risk_formula))
y = (probabilities > 0.84).astype(int)

feature_columns = ['working_capital_ratio', 'retained_earnings_ratio', 'ebitda_ratio', 'sentiment_score']
df = pd.DataFrame(X_combined, columns=feature_columns)
X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.25, random_state=42, stratify=y)

bankruptcy_pipeline = Pipeline([
    ('scaler', RobustScaler()),
    ('resampler', SMOTE(random_state=42, sampling_strategy=0.35)),
    ('classifier', LogisticRegression(C=0.85, class_weight='balanced', random_state=42))
])

print('Training pipeline components on imbalanced metrics...')
bankruptcy_pipeline.fit(X_train, y_train)
joblib.dump(bankruptcy_pipeline, 'pipeline.joblib')
print('SUCCESS: Operational pipeline binary compiled as pipeline.joblib')
