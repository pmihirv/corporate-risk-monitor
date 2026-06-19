from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse # Added StreamingResponse here
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator
import joblib
import pandas as pd
import os

from financial_parser import extract_and_calculate_ratios
# Import our new report generator function
from report_generator import generate_pdf_report 

app = FastAPI(
    title="Enterprise Insolvency Risk Monitor API",
    version="3.1"
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
    print("✅ High-credibility Pipeline loaded successfully.")
except Exception as e:
    print(f"❌ Failed loading pipeline: {e}")
    model = None

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
    def validate_sentiment(cls, v: float) -> float:
        if not (-1.0 <= v <= 1.0):
            raise ValueError('Sentiment must be between -1.0 and +1.0')
        return v

# PDF Query Parameters Contract definition
class PDFReportRequest(BaseModel):
    company_name: str
    altman_z_score: float
    financial_risk_zone: str
    ml_risk_probability: float
    model_classification: str
    x1: float
    x2: float
    x3: float
    x4: float
    x5: float
    sentiment: float

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/ticker/{symbol}")
def get_ticker_data(symbol: str):
    try:
        ratios = extract_and_calculate_ratios(symbol)
        return ratios
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_risk(payload: RiskEvaluationRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model binary not loaded.")
        
    try:
        input_data = pd.DataFrame([{
            'working_capital_ta': payload.working_capital_ta,
            'retained_earnings_ta': payload.retained_earnings_ta,
            'ebitda_ta': payload.ebitda_ta,
            'market_cap_tl': payload.market_cap_tl,
            'sales_ta': payload.sales_ta,
            'sentiment': payload.sentiment
        }])

        if payload.firm_type == "private":
            z_score = (0.717 * payload.working_capital_ta + 0.847 * payload.retained_earnings_ta + 
                       3.107 * payload.ebitda_ta + 0.420 * payload.market_cap_tl + 0.998 * payload.sales_ta)
            zone = "Safe Zone" if z_score >= 2.90 else ("Grey Zone" if z_score >= 1.23 else "Distress Zone")
        else:
            z_score = (1.2 * payload.working_capital_ta + 1.4 * payload.retained_earnings_ta + 
                       3.3 * payload.ebitda_ta + 0.6 * payload.market_cap_tl + 1.0 * payload.sales_ta)
            zone = "Safe Zone" if z_score >= 2.99 else ("Grey Zone" if z_score >= 1.81 else "Distress Zone")
        
        probabilities = model.predict_proba(input_data)[0]
        risk_probability = float(probabilities[1])
        
        return {
            "altman_z_score": round(z_score, 2),
            "financial_risk_zone": zone,
            "ml_risk_probability": round(risk_probability * 100, 2),
            "model_classification": "Distress / Risk" if risk_probability >= 0.50 else "No Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {str(e)}")

# The Dynamic PDF Document Generator Endpoint
@app.post("/api/report/download")
def download_report(payload: PDFReportRequest):
    try:
        pdf_stream = generate_pdf_report(payload.model_dump())
        filename = f"Risk_Report_{payload.company_name.replace(' ', '_')}.pdf"
        return StreamingResponse(
            pdf_stream,
            media_type="application/pdf",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))