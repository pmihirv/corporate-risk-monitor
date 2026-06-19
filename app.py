from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import List, Optional, Dict
import joblib
import pandas as pd
import os

from financial_parser import extract_and_calculate_ratios
from report_generator import generate_pdf_report 

app = FastAPI(title="Enterprise Risk Monitor API", version="4.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

current_dir = os.path.dirname(os.path.abspath(__file__))
templates = Jinja2Templates(directory=os.path.join(current_dir, "templates"))
model = joblib.load("pipeline.joblib")

class RiskEvaluationRequest(BaseModel):
    working_capital_ta: float; retained_earnings_ta: float; ebitda_ta: float; market_cap_tl: float; sales_ta: float; sentiment: float; firm_type: str = "public"

class PDFReportRequest(BaseModel):
    ticker_symbol: str; company_name: str; altman_z_score: float; financial_risk_zone: str; ml_risk_probability: float; model_classification: str; x1: float; x2: float; x3: float; x4: float; x5: float; sentiment: float; history: Optional[List[Dict[str, float]]] = None

@app.get("/", response_class=HTMLResponse)
def home(request: Request): return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/ticker/{symbol}")
def get_ticker_data(symbol: str):
    try: return extract_and_calculate_ratios(symbol)
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict")
def predict_risk(payload: RiskEvaluationRequest):
    try:
        input_data = pd.DataFrame([{"working_capital_ta": payload.working_capital_ta, "retained_earnings_ta": payload.retained_earnings_ta, "ebitda_ta": payload.ebitda_ta, "market_cap_tl": payload.market_cap_tl, "sales_ta": payload.sales_ta, "sentiment": payload.sentiment}])
        z_score = (1.2 * payload.working_capital_ta + 1.4 * payload.retained_earnings_ta + 3.3 * payload.ebitda_ta + 0.6 * payload.market_cap_tl + 1.0 * payload.sales_ta)
        zone = "Safe Zone" if z_score >= 2.99 else ("Grey Zone" if z_score >= 1.81 else "Distress Zone")
        prob = float(model.predict_proba(input_data)[0][1])
        return {"altman_z_score": round(z_score, 2), "financial_risk_zone": zone, "ml_risk_probability": round(prob * 100, 2), "model_classification": "Distress / Risk" if prob >= 0.50 else "No Risk"}
    except Exception as e: raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/report/download")
def download_report(payload: PDFReportRequest):
    try:
        pdf = generate_pdf_report(payload.model_dump())
        return StreamingResponse(pdf, media_type="application/pdf", headers={"Content-Disposition": f"attachment; filename=Risk_Report.pdf"})
    except Exception as e: raise HTTPException(status_code=500, detail=str(e))
