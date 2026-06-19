from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import joblib

app = FastAPI(title='Enterprise Insolvency Risk Monitor')
templates = Jinja2Templates(directory='templates')
model_pipeline = joblib.load('pipeline.joblib')

@app.get('/', response_class=HTMLResponse)
async def render_dashboard(request: Request):
    return templates.TemplateResponse(request=request, name='index.html', context={'prediction_text': None})

@app.post('/evaluate', response_class=HTMLResponse)
async def run_risk_evaluation(
    request: Request,
    working_capital: float = Form(...),
    retained_earnings: float = Form(...),
    ebitda: float = Form(...),
    sentiment_score: float = Form(...)
):
    input_payload = pd.DataFrame([[working_capital, retained_earnings, ebitda, sentiment_score]], 
                                 columns=['working_capital_ratio', 'retained_earnings_ratio', 'ebitda_ratio', 'sentiment_score'])
    predicted_class = model_pipeline.predict(input_payload)[0]
    predicted_probability = model_pipeline.predict_proba(input_payload)[0][1]
    
    verdict_text = 'CRITICAL INSOLVENCY PROFILE DETECTED' if predicted_class == 1 else 'STABLE / SOLID SOLVENCY MARGINS'
    color_alert = '#dc3545' if predicted_class == 1 else '#198754'
        
    evaluation_metrics = {'verdict': verdict_text, 'probability': f'{predicted_probability * 100:.2f}%', 'theme': color_alert}
    return templates.TemplateResponse(request=request, name='index.html', context={'prediction_text': evaluation_metrics})
