from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="Enterprise Insolvency Risk Monitor API",
    description="Modified Altman Z-Score + FinBERT Sentiment Pipeline",
    version="2.0"
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the newly trained ML pipeline binary
try:
    model = joblib.load("pipeline.joblib")
except Exception as e:
    print(f"Error loading model binary: {e}")
    model = None

# 1. Upgraded Request Schema mapping all 5 Altman Variables + Sentiment
class RiskEvaluationRequest(BaseModel):
    working_capital_ta: float = Field(..., description="X1: Working Capital / Total Assets")
    retained_earnings_ta: float = Field(..., description="X2: Retained Earnings / Total Assets")
    ebitda_ta: float = Field(..., description="X3: EBITDA / Total Assets")
    market_cap_tl: float = Field(..., description="X4: Market Cap / Total Liabilities")
    sales_ta: float = Field(..., description="X5: Sales / Total Assets")
    sentiment: float = Field(..., description="Alternative Textual Sentiment Score (-1.0 to 1.0)")

@app.get("/")
def home():
    return {
        "status": "online",
        "framework": "Modified Altman Z-Score (5 Ratios) + NLP Sentiment",
        "documentation": "/docs"
    }

@app.post("/predict")
def predict_risk(payload: RiskEvaluationRequest):
    if not model:
        raise HTTPException(status_code=500, detail="ML model binary not compiled or loaded.")
    
    # 2. Extract inputs into DataFrame format matching training features
    input_data = pd.DataFrame([{
        'working_capital_ta': payload.working_capital_ta,
        'retained_earnings_ta': payload.retained_earnings_ta,
        'ebitda_ta': payload.ebitda_ta,
        'market_cap_tl': payload.market_cap_tl,
        'sales_ta': payload.sales_ta,
        'sentiment': payload.sentiment
    }])
    
    try:
        # 3. Calculate Classic Altman Z-Score baseline for public firms
        # Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5
        z_score = (
            1.2 * payload.working_capital_ta +
            1.4 * payload.retained_earnings_ta +
            3.3 * payload.ebitda_ta +
            0.6 * payload.market_cap_tl +
            1.0 * payload.sales_ta
        )
        
        # 4. Compute ML dynamic classification probability
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        risk_probability = float(probabilities[1])
        
        # Determine strict financial risk band zone based on standard Z-Score thresholds
        if z_score >= 2.99:
            zone = "Safe Zone"
        elif z_score >= 1.81:
            zone = "Grey Zone"
        else:
            zone = "Distress Zone"
            
        return {
            "altman_z_score": round(z_score, 2),
            "financial_risk_zone": zone,
            "ml_risk_probability": round(risk_probability * 100, 2),
            "model_classification": "Distress / Risk" if prediction == 1 else "No Risk"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating metrics: {str(e)}")