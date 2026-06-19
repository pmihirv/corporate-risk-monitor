from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
import traceback
import os

app = FastAPI(
    title="Enterprise Insolvency Risk Monitor API",
    description="Modified Altman Z-Score + FinBERT Sentiment Pipeline",
    version="2.0"
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
except Exception as e:
    print(f"Error loading model binary: {e}")
    model = None

class RiskEvaluationRequest(BaseModel):
    working_capital_ta: float = Field(..., description="X1: Working Capital / Total Assets")
    retained_earnings_ta: float = Field(..., description="X2: Retained Earnings / Total Assets")
    ebitda_ta: float = Field(..., description="X3: EBITDA / Total Assets")
    market_cap_tl: float = Field(..., description="X4: Market Cap / Total Liabilities")
    sales_ta: float = Field(..., description="X5: Sales / Total Assets")
    sentiment: float = Field(..., description="Alternative Textual Sentiment Score (-1.0 to 1.0)")

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    try:
        # Fixed: Explicitly passing parameters resolves the unhashable type dictionary bug
        return templates.TemplateResponse(request=request, name="index.html")
    except Exception as e:
        print("\n" + "="*50 + "\nCRITICAL FRONTEND RENDER ERROR:\n" + "="*50)
        traceback.print_exc()
        print("="*50 + "\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict")
def predict_risk(payload: RiskEvaluationRequest):
    if not model:
        raise HTTPException(status_code=500, detail="ML model binary not compiled or loaded.")
    
    input_data = pd.DataFrame([{
        'working_capital_ta': payload.working_capital_ta,
        'retained_earnings_ta': payload.retained_earnings_ta,
        'ebitda_ta': payload.ebitda_ta,
        'market_cap_tl': payload.market_cap_tl,
        'sales_ta': payload.sales_ta,
        'sentiment': payload.sentiment
    }])
    
    try:
        z_score = (
            1.2 * payload.working_capital_ta +
            1.4 * payload.retained_earnings_ta +
            3.3 * payload.ebitda_ta +
            0.6 * payload.market_cap_tl +
            1.0 * payload.sales_ta
        )
        
        prediction = int(model.predict(input_data)[0])
        probabilities = model.predict_proba(input_data)[0]
        risk_probability = float(probabilities[1])
        
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