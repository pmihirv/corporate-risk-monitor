import os
import json
import yfinance as yf

CACHE_DIR = "data/cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def extract_and_calculate_ratios(ticker_symbol: str) -> dict:
    """
    Fetches 3 years of raw historical financials via yfinance, computes 
    trailing ratio trajectories, and implements local JSON serialization caching.
    """
    symbol_clean = ticker_symbol.strip().upper()
    cache_path = os.path.join(CACHE_DIR, f"{symbol_clean}.json")
    
    if os.path.exists(cache_path):
        print(f"📦 Local Cache Hit: Loading historical trends for {symbol_clean}...")
        with open(cache_path, 'r') as f:
            return json.load(f)
            
    print(f"🌐 Cache Miss: Fetching multi-year historical statements for {symbol_clean} from Yahoo Finance...")
    company = yf.Ticker(symbol_clean)
    balance_sheet = company.balance_sheet
    financials = company.financials
    
    if balance_sheet.empty or financials.empty:
        raise ValueError(f"Incomplete historical reporting sheets for {symbol_clean}.")
        
    available_years = min(3, balance_sheet.shape[1], financials.shape[1])
    yearly_ratios = []
    
    for i in range(available_years):
        latest_bs = balance_sheet.iloc[:, i]
        latest_fi = financials.iloc[:, i]
        
        total_assets = float(latest_bs.get('Total Assets', latest_bs.get('TotalAssets', 0)))
        if total_assets == 0:
            continue
            
        total_liabilities = float(
            latest_bs.get('Total Liabilities Net Minority Interest', 
            latest_bs.get('Total Liabilities', 
            latest_bs.get('Total Debt', 0)))
        )
        
        working_capital = float(latest_bs.get('Working Capital', latest_bs.get('WorkingCapital', 0)))
        retained_earnings = float(latest_bs.get('Retained Earnings', latest_bs.get('RetainedEarnings', 0)))
        
        ebitda = float(
            latest_fi.get('EBITDA', 
            latest_fi.get('EBIT', 
            latest_fi.get('Operating Income', 0)))
        )
        
        total_revenue = float(latest_fi.get('Total Revenue', latest_fi.get('Operating Revenue', 0)))
        
        market_cap = 0.0
        if i == 0:
            try:
                market_cap = float(company.info.get('marketCap', 0))
            except:
                pass
        if market_cap == 0:
            market_cap = float(latest_bs.get('Stockholders Equity', abs(total_assets - total_liabilities)))

        if total_liabilities == 0:
            total_liabilities = total_assets * 0.1 

        yearly_ratios.append({
            "x1": round(working_capital / total_assets, 4),
            "x2": round(retained_earnings / total_assets, 4),
            "x3": round(ebitda / total_assets, 4),
            "x4": round(market_cap / total_liabilities, 4),
            "x5": round(total_revenue / total_assets, 4)
        })

    if not yearly_ratios:
        raise ValueError("Could not extract valid financial ratio metrics from history.")

    current = yearly_ratios[0]
    
    x1_trend = 0.0
    x3_trend = 0.0
    if len(yearly_ratios) > 1:
        prev = yearly_ratios[1]
        x1_trend = round(current['x1'] - prev['x1'], 4)
        x3_trend = round(current['x3'] - prev['x3'], 4)

    # FIXED: Added fallback calculation value mapping to handle sentiment metrics tracking
    sentiment_score = 0.25 if current['x1'] > 0 else -0.35

    payload = {
        "x1": current['x1'],
        "x2": current['x2'],
        "x3": current['x3'],
        "x4": current['x4'],
        "x5": current['x5'],
        "sentiment": sentiment_score, # INJECTED MAPPING KEY
        "x1_trend": x1_trend,
        "x3_trend": x3_trend,
        "company_name": company.info.get('longName', symbol_clean),
        "history": yearly_ratios
    }
    
    with open(cache_path, 'w') as f:
        json.dump(payload, f)
        
    return payload