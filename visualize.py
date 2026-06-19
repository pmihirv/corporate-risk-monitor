import os
import json
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap

print("=" * 70)
print("EXECUTION MODULE: ROBUST RESEARCH VISUALIZATION ENGINE")
print("=" * 70)

def generate_research_plots(ticker_symbol: str):
    symbol_clean = ticker_symbol.strip().upper()
    cache_path = f"data/cache/{symbol_clean}.json"
    output_dir = "data/visuals"
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"❌ Cache vector missing for {symbol_clean}. Run a parse check first.")
        
    with open(cache_path, 'r') as f:
        data = json.load(f)

    print(f"📦 Loaded cached parameters for {symbol_clean}. Building structural visuals...")

    # Plot 1: Multi-Year Financial Trajectory Trends
    history = data.get('history', [])
    if len(history) >= 1:
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid' in plt.style.available else 'default')
        fig, ax = plt.subplots(figsize=(7, 4))
        
        chronological_history = history[::-1]
        years = [f"Y-{len(chronological_history)-1-i}" if (len(chronological_history)-1-i) > 0 else "Current" for i in range(len(chronological_history))]
        
        x1_vals = [h['x1'] for h in chronological_history]
        x3_vals = [h['x3'] for h in chronological_history]
        x5_vals = [h['x5'] for h in chronological_history]
        
        ax.plot(years, x1_vals, marker='o', linewidth=2, color='#0284c7', label='X1: Working Capital/TA')
        ax.plot(years, x3_vals, marker='s', linewidth=2, color='#10b981', label='X3: EBITDA/TA')
        ax.plot(years, x5_vals, marker='^', linewidth=2, color='#f59e0b', label='X5: Asset Turnover')
        
        ax.set_title(f"Trailing Financial Ratio Trajectories: {data['company_name']}", fontsize=11, fontweight='bold', pad=10)
        ax.set_ylabel("Ratio Scale Matrix Value", fontsize=9)
        ax.legend(loc='best', fontsize=8)
        
        trend_png_path = os.path.join(output_dir, f"{symbol_clean}_trends.png")
        plt.tight_layout()
        plt.savefig(trend_png_path, dpi=150)
        plt.close()
        print(f"📈 Trajectory trend lines exported to: '{trend_png_path}'")

    # Plot 2: Localized SHAP Decision Boundary Attribution Map
    try:
        pipeline = joblib.load("pipeline.joblib")
        
        # Match names exactly to the training features array layout
        input_vector = pd.DataFrame([{
            'working_capital_ta': float(data['x1']),
            'retained_earnings_ta': float(data['x2']),
            'ebitda_ta': float(data['x3']),
            'market_cap_tl': float(data['x4']),
            'sales_ta': float(data['x5']),
            'sentiment': float(data['sentiment'])
        }])
        
        # Standardize feature ordering to prevent XGBoost alignment mismatch errors
        features_ordered = ['working_capital_ta', 'retained_earnings_ta', 'ebitda_ta', 'market_cap_tl', 'sales_ta', 'sentiment']
        input_vector = input_vector[features_ordered]
        
        scaled_values = pipeline.named_steps['scaler'].transform(input_vector)
        classifier_model = pipeline.named_steps['xgb']
        
        explainer = shap.TreeExplainer(classifier_model)
        raw_shap_values = explainer.shap_values(scaled_values)
        
        if isinstance(raw_shap_values, list):
            attributions = raw_shap_values[1][0] if len(raw_shap_values) > 1 else raw_shap_values[0][0]
        elif len(raw_shap_values.shape) == 3:
            attributions = raw_shap_values[0, :, 1]
        elif len(raw_shap_values.shape) == 2 and raw_shap_values.shape[0] == 1:
            attributions = raw_shap_values[0]
        else:
            attributions = raw_shap_values

        feature_display_names = ['X1: Working Cap/TA', 'X2: Retained Earn/TA', 'X3: EBITDA/TA', 'X4: Market Cap/TL', 'X5: Sales/TA', 'Sentiment Proxy']
        
        fig, ax = plt.subplots(figsize=(7, 4))
        y_pos = np.arange(len(feature_display_names))
        
        bar_colors = ['#10b981' if val < 0 else '#ef4444' for val in attributions]
        
        ax.barh(y_pos, attributions, align='center', color=bar_colors, alpha=0.85, edgecolor='#cbd5e1', height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_display_names, fontsize=9)
        ax.axvline(0, color='#64748b', linewidth=0.8, linestyle='--')
        
        ax.set_xlabel("SHAP Impact Margin (Log-Odds Attribution Scale)", fontsize=9)
        ax.set_title(f"Model Feature Risk Contribution: {symbol_clean}", fontsize=11, fontweight='bold', pad=10)
        
        shap_png_path = os.path.join(output_dir, f"{symbol_clean}_shap.png")
        plt.tight_layout()
        plt.savefig(shap_png_path, dpi=150)
        plt.close()
        print(f"🔮 Tree explainer risk matrices written to: '{shap_png_path}'")
        
    except Exception as e:
        print(f"❌ SHAP Error Diagnostics: {str(e)}")

    print("=" * 70)

if __name__ == "__main__":
    import sys
    target = sys.argv[1] if len(sys.argv) > 1 else "NVDA"
    generate_research_plots(target)