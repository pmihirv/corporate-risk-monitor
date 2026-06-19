from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import traceback
import os
import yfinance as ticker_fetcher

app = FastAPI(
    title="Enterprise Insolvency Risk Monitor API",
    description="Deterministic Altman Math + Robust Pipeline Alignment",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

current_dir = os.path.dirname(os.path.abspath(__file__))
templates_dir = os.path.join(current_dir, "templates")
templates = Jinja2Templates(directory=templates_dir)

try:
    model = joblib.load("pipeline.joblib")
    print("✅ High-credibility model pipeline artifact loaded.")
except Exception as e:
    print(f"❌ Error loading pipeline artifact: {e}")
    model = None

# PRODUCTION RECOMMENDATION: Pydantic Input Boundary Validation Checkers
class RiskEvaluationRequest(BaseModel):
    working_capital_ta: float
    retained_earnings_ta: float
    ebitda_ta: float
    market_cap_tl: float
    sales_ta: float
    sentiment: float
    firm_type: str = "public"

    @field_validator('sentiment')
    @classmethod
    def validate_sentiment_range(cls, v: float) -> float:
        if not (-1.0 <= v <= 1.0):
            raise ValueError('Sentiment parameters must fall strictly between -1.0 and +1.0')
        return v

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/ticker/{symbol}")
def get_ticker_data(symbol: str):
    try:
        company = ticker_fetcher.Ticker(symbol)
        balance_sheet = company.balance_sheet
        financials = company.financials
        
        if balance_sheet.empty or financials.empty:
            raise HTTPException(status_code=404, detail="Financial sheets unavailable or delisted for this ticker.")
            
        latest_bs = balance_sheet.iloc[:, 0]
        latest_fi = financials.iloc[:, 0]
        
        # PRODUCTION RECOMMENDATION: Prioritized Key Fallback Matrix Architecture
        total_assets = float(
            latest_bs.get('Total Assets', 0) or 
            latest_bs.get('TotalAssets', 0) or 0
        )
        
        total_liabilities = float(
            latest_bs.get('Total Liabilities Net Minority Interest', 0) or 
            latest_bs.get('Total Liabilities', 0) or 
            latest_bs.get('Total Debt', 0) or 0
        )
        
        working_capital = float(
            latest_bs.get('Working Capital', 0) or 
            latest_bs.get('WorkingCapital', 0) or 0
        )
        
        retained_earnings = float(
            latest_bs.get('Retained Earnings', 0) or 
            latest_bs.get('RetainedEarnings', 0) or 0
        )
        
        # Fallbacks to capture shifting structural definitions of operational income
        ebitda = float(
            latest_fi.get('EBITDA', 0) or 
            latest_fi.get('EBIT', 0) or 
            latest_fi.get('Operating Income', 0) or 
            latest_fi.get('Gross Profit', 0) or 0
        )
        
        total_revenue = float(
            latest_fi.get('Total Revenue', 0) or 
            latest_fi.get('Operating Revenue', 0) or 0
        )
        
        if total_assets == 0:
            raise HTTPException(status_code=400, detail="Incomplete corporate reporting profile found (Zero Assets).")
            
        # Secure dynamic market capitalization matrix evaluations
        market_cap = 0.0
        try:
            market_cap = float(company.info.get('marketCap', 0) or company.info.get('enterpriseValue', 0) or 0)
        except:
            pass
            
        if market_cap == 0:
            equity = float(latest_bs.get('Stockholders Equity', 0) or latest_bs.get('Total Equity Gross Minority Interest', 0) or 0)
            market_cap = equity if equity != 0 else abs(total_assets - total_liabilities)

        if total_liabilities == 0:
            total_liabilities = total_assets * 0.4  # Defensive risk proxy calculation fallback
            
        return {
            "x1": round(working_capital / total_assets, 4),
            "x2": round(retained_earnings / total_assets, 4),
            "x3": round(ebitda / total_assets, 4),
            "x4": round(market_cap / total_liabilities, 4),
            "x5": round(total_revenue / total_assets, 4),
            "company_name": company.info.get('longName', symbol.upper())
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Data parser error: {str(e)}")

@app.post("/predict")
def predict_risk(payload: RiskEvaluationRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model binary unavailable.")
        
    try:
        input_data = pd.DataFrame([{
            'working_capital_ta': payload.working_capital_ta,
            'retained_earnings_ta': payload.retained_earnings_ta,
            'ebitda_ta': payload.ebitda_ta,
            'market_cap_tl': payload.market_cap_tl,
            'sales_ta': payload.sales_ta,
            'sentiment': payload.sentiment
        }])

        # Deterministic Corporate Baseline calculations
        if payload.firm_type == "private":
            z_score = (0.717 * payload.working_capital_ta + 0.847 * payload.retained_earnings_ta + 
                       3.107 * payload.ebitda_ta + 0.420 * payload.market_cap_tl + 0.998 * payload.sales_ta)
            zone = "Safe Zone" if z_score >= 2.90 else ("Grey Zone" if z_score >= 1.23 else "Distress Zone")
        else:
            z_score = (1.2 * payload.working_capital_ta + 1.4 * payload.retained_earnings_ta + 
                       3.3 * payload.ebitda_ta + 0.6 * payload.market_cap_tl + 1.0 * payload.sales_ta)
            zone = "Safe Zone" if z_score >= 2.99 else ("Grey Zone" if z_score >= 1.81 else "Distress Zone")
        
        # Pure Machine Learning Pipeline Inference (Pipeline handles RobustScaling seamlessly)
        probabilities = model.predict_proba(input_data)[0]
        risk_probability = float(probabilities[1])
        
        return {
            "altman_z_score": round(z_score, 2),
            "financial_risk_zone": zone,
            "ml_risk_probability": round(risk_probability * 100, 2),
            "model_classification": "Distress / Risk" if risk_probability >= 0.50 else "No Risk"
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")