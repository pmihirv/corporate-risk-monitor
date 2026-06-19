import yfinance as yf

def extract_and_calculate_ratios(ticker_symbol: str) -> dict:
    """
    Fetches raw financial statements via yfinance and computes 
    the precise 5 Altman Z-Score ratios with dynamic fallbacks.
    """
    company = yf.Ticker(ticker_symbol)
    balance_sheet = company.balance_sheet
    financials = company.financials
    
    if balance_sheet.empty or financials.empty:
        raise ValueError("Financial statements are empty or delisted for this symbol.")
        
    # Extract the most recent fiscal year data column
    latest_bs = balance_sheet.iloc[:, 0]
    latest_fi = financials.iloc[:, 0]
    
    # 1. Base Variables with Multi-Key Fallbacks
    total_assets = float(latest_bs.get('Total Assets', latest_bs.get('TotalAssets', 0)))
    if total_assets == 0:
        raise ValueError("Total Assets is zero; cannot compute structural ratios.")
        
    total_liabilities = float(
        latest_bs.get('Total Liabilities Net Minority Interest', 
        latest_bs.get('Total Liabilities', 
        latest_bs.get('Total Debt', 0)))
    )
    
    working_capital = float(latest_bs.get('Working Capital', latest_bs.get('WorkingCapital', 0)))
    retained_earnings = float(latest_bs.get('Retained Earnings', latest_bs.get('RetainedEarnings', 0)))
    
    # 2. Operational Performance Fallbacks
    ebitda = float(
        latest_fi.get('EBITDA', 
        latest_fi.get('EBIT', 
        latest_fi.get('Operating Income', 0)))
    )
    
    total_revenue = float(latest_fi.get('Total Revenue', latest_fi.get('Operating Revenue', 0)))
    
    # 3. Market Valuation / Equity Fallbacks
    market_cap = 0.0
    try:
        market_cap = float(company.info.get('marketCap', 0))
    except:
        pass
        
    if market_cap == 0:
        # Fallback to Book Value of Equity if Market Cap is missing (or for private profiles)
        market_cap = float(latest_bs.get('Stockholders Equity', total_assets - total_liabilities))

    # Defensive guard to avoid ZeroDivisionError on debt-free companies
    if total_liabilities == 0:
        total_liabilities = total_assets * 0.1 

    # 4. Compute the Precise 5 Altman Ratios
    return {
        "x1": round(working_capital / total_assets, 4),
        "x2": round(retained_earnings / total_assets, 4),
        "x3": round(ebitda / total_assets, 4),
        "x4": round(market_cap / total_liabilities, 4),
        "x5": round(total_revenue / total_assets, 4),
        "company_name": company.info.get('longName', ticker_symbol.upper())
    }