import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler

print("=" * 70)
print("EXECUTION MODULE: UNSUPERVISED BEHAVIORAL PATTERN & CLUSTERING ENGINE")
print("=" * 70)

# Establish local storage directory bounds
os.makedirs("models", exist_ok=True)
csv_path = "data/raw/bankruptcy_data.csv"

def train_unsupervised_patterns():
    """
    Trains an Isolation Forest for outlier/fraud flagging and K-Means for 
    historical corporate peer clustering based on real-world macro profiles.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing base historical dataset at '{csv_path}' to train patterns.")
        
    df = pd.read_csv(csv_path)
    features = ['working_capital_ta', 'retained_earnings_ta', 'ebitda_ta', 'market_cap_tl', 'sales_ta']
    
    # Clean matrices
    X = df[features].dropna()
    
    # Scale data uniformly using RobustScaler matching your supervised pipeline
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("🌲 Fitting Isolation Forest anomaly detection vectors...")
    # contamination=0.03 means we explicitly expect roughly 3% of firms to display extreme outlier profiles
    iso_forest = IsolationForest(contamination=0.03, random_state=42)
    iso_forest.fit(X_scaled)
    
    print("🤖 Compiling K-Means micro-clustering matrices (4 Corporate Archetypes)...")
    # Group firms into 4 distinct operational archetypes
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    
    # Serialize patterns artifacts
    joblib.dump(scaler, "models/pattern_scaler.joblib")
    joblib.dump(iso_forest, "models/anomaly_detector.joblib")
    joblib.dump(kmeans, "models/peer_clusterer.joblib")
    print("🎯 Behavioral profile models written out safely to 'models/' directory.")

def diagnose_company_pattern(ticker_symbol: str):
    """
    Loads cached ticker statements and evaluates its anomaly flags and behavioral cluster peer groups.
    """
    symbol_clean = ticker_symbol.strip().upper()
    cache_path = f"data/cache/{symbol_clean}.json"
    
    if not os.path.exists(cache_path):
        print(f"❌ No cached statements found for {symbol_clean}. Fetch data first via app parser.")
        return
        
    with open(cache_path, 'r') as f:
        data = json.load(f)
        
    # Check if models are built; if not, compile them instantly
    if not os.path.exists("models/anomaly_detector.joblib"):
        train_unsupervised_patterns()
        
    scaler = joblib.load("models/pattern_scaler.joblib")
    iso_forest = joblib.load("models/anomaly_detector.joblib")
    kmeans = joblib.load("models/peer_clusterer.joblib")
    
    # FIXED: Replaced raw numpy array with a named DataFrame matching training feature headers to silence sklearn warning
    features_ordered = ['working_capital_ta', 'retained_earnings_ta', 'ebitda_ta', 'market_cap_tl', 'sales_ta']
    input_vector = pd.DataFrame([{
        'working_capital_ta': float(data['x1']),
        'retained_earnings_ta': float(data['x2']),
        'ebitda_ta': float(data['x3']),
        'market_cap_tl': float(data['x4']),
        'sales_ta': float(data['x5'])
    }])[features_ordered]
    
    # Execute scaling transformations smoothly
    scaled_vector = scaler.transform(input_vector)
    
    # Extract Anomaly Status Vector (-1 = Outlier, 1 = Normal structural range)
    anomaly_status = iso_forest.predict(scaled_vector)[0]
    
    # Predict Core Cluster Grouping
    cluster_id = kmeans.predict(scaled_vector)[0]
    
    # Human-readable mapping definitions of unsupervised corporate clusters
    cluster_archetypes = {
        0: "Highly Leveraged Mature Enterprise (High debt, standard operational turnover)",
        1: "Distressed Operations Cell (Asset erosion, capital decay velocity risk)",
        2: "Capital-Protected Compounder (Solid liquidity cushion, high retained earnings buffer)",
        3: "Hyper-Growth Scaler / Capital Outlier (Massive market cap relative to baseline liabilities)"
    }
    
    print(f"\n📋 BEHAVIORAL PATTERN DIAGNOSTIC DOSSIER FOR: {symbol_clean}")
    print("-" * 65)
    
    if anomaly_status == -1:
        print("🚨 ANOMALY ALERT: This enterprise displays extreme outlier behavior!")
        print("   Metrics deviate wildly from historical standard distributions. Forensic inspection advised.")
    else:
        print("✅ STRUCTURAL ALIGNMENT: Financial distribution matches normal systemic boundary spreads.")
        
    print(f"Group Archetype Profile: {cluster_archetypes.get(cluster_id, 'Unknown Archetype Cluster')}")
    print("-" * 65)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_unsupervised_patterns()
    else:
        target = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
        diagnose_company_pattern(target)