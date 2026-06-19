import os
import io
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_live_plots(data: dict, ticker_token: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use("default")
    
    history = data.get("history") or [{"x1": data["x1"], "x3": data["x3"], "x5": data["x5"]}]
    fig, ax = plt.subplots(figsize=(7, 4))
    chronological_history = history[::-1]
    years = [f"Y-{len(chronological_history)-1-i}" if (len(chronological_history)-1-i) > 0 else "Current" for i in range(len(chronological_history))]
    
    ax.plot(years, [h.get("x1", data["x1"]) for h in chronological_history], marker="o", linewidth=2, color="#0284c7", label="X1: Working Cap/TA")
    ax.plot(years, [h.get("x3", data["x3"]) for h in chronological_history], marker="s", linewidth=2, color="#10b981", label="X3: EBITDA/TA")
    ax.plot(years, [h.get("x5", data["x5"]) for h in chronological_history], marker="^", linewidth=2, color="#f59e0b", label="X5: Asset Turnover")
    ax.set_title(f"Financial Ratio Trajectories: {ticker_token}", fontsize=11, fontweight="bold")
    ax.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{ticker_token}_trends.png"), dpi=150)
    plt.close()

    try:
        pipeline = joblib.load("pipeline.joblib")
        input_vector = pd.DataFrame([{
            "working_capital_ta": float(data["x1"]), "retained_earnings_ta": float(data["x2"]),
            "ebitda_ta": float(data["x3"]), "market_cap_tl": float(data["x4"]),
            "sales_ta": float(data["x5"]), "sentiment": float(data["sentiment"])
        }])
        input_vector = input_vector[["working_capital_ta", "retained_earnings_ta", "ebitda_ta", "market_cap_tl", "sales_ta", "sentiment"]]
        scaled_values = pipeline.named_steps["scaler"].transform(input_vector)
        explainer = shap.TreeExplainer(pipeline.named_steps["xgb"])
        raw_shap_values = explainer.shap_values(scaled_values)
        
        attributions = raw_shap_values[0] if len(raw_shap_values.shape) == 2 else raw_shap_values
        feature_display_names = ["X1: Working Cap", "X2: Retained Earn", "X3: EBITDA", "X4: Market Cap", "X5: Sales", "Sentiment"]
        
        fig, ax = plt.subplots(figsize=(7, 4))
        y_pos = np.arange(len(feature_display_names))
        bar_colors = ["#10b981" if val < 0 else "#ef4444" for val in attributions]
        ax.barh(y_pos, attributions, align="center", color=bar_colors, edgecolor="#cbd5e1", height=0.6)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_display_names, fontsize=9)
        ax.axvline(0, color="#64748b", linewidth=0.8, linestyle="--")
        ax.set_title(f"Model Feature Risk Contribution: {ticker_token}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{ticker_token}_shap.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"SHAP Error bypass: {e}")

def generate_pdf_report(data: dict) -> io.BytesIO:
    ticker_token = str(data.get("ticker_symbol", "CUSTOM")).strip().upper()
    visuals_dir = "data/visuals"
    generate_live_plots(data, ticker_token, visuals_dir)
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40)
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle("DocTitle", parent=styles["Heading1"], fontSize=20, leading=24, textColor=colors.HexColor("#0f172a"))
    section_style = ParagraphStyle("SecTitle", parent=styles["Heading2"], fontSize=12, leading=16, textColor=colors.HexColor("#0284c7"), spaceBefore=10, spaceAfter=5)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=9, leading=13, textColor=colors.HexColor("#334155"))

    story = [
        Paragraph("EXECUTIVE CREDIT RISK EVALUATION REPORT", title_style),
        Paragraph(f"Target Entity Asset Profile: <b>{data.get('company_name', 'Unknown')} ({ticker_token})</b>", body_style),
        Spacer(1, 10),
        Paragraph("1. Core Predictive Risk Analytics", section_style)
    ]
    
    zone = data.get("financial_risk_zone", "Grey Zone")
    zone_color = "#10b981" if "Safe" in zone else ("#f59e0b" if "Grey" in zone else "#ef4444")
    
    summary_table_data = [
        [Paragraph("<b>Altman Z-Score Matrix</b>", body_style), Paragraph(f"<font color='{zone_color}'><b>{data.get('altman_z_score', '0.0')} ({zone})</b></font>", body_style)],
        [Paragraph("<b>XGBoost Default Probability</b>", body_style), Paragraph(f"<b>{data.get('ml_risk_probability', '0.0')}% Probability</b>", body_style)],
        [Paragraph("<b>Model Classification Decision</b>", body_style), Paragraph(f"<b>{data.get('model_classification', 'No Risk')}</b>", body_style)]
    ]
    t_summary = Table(summary_table_data, colWidths=[240, 280])
    t_summary.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("PADDING", (0, 0), (-1, -1), 6)
    ]))
    story.append(t_summary)
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("2. Deep-Dive Explainability & Trajectory Analysis Charts", section_style))
    
    t_path = os.path.join(visuals_dir, f"{ticker_token}_trends.png")
    s_path = os.path.join(visuals_dir, f"{ticker_token}_shap.png")
    cells = []
    cells.append(Image(t_path, width=250, height=142) if os.path.exists(t_path) else Paragraph("Pending...", body_style))
    cells.append(Image(s_path, width=250, height=142) if os.path.exists(s_path) else Paragraph("Pending...", body_style))
    
    t_vis = Table([cells], colWidths=[260, 260])
    story.append(t_vis)
    
    story.append(Spacer(1, 10))
    story.append(Paragraph("3. Extracted Financial Ratio Vectors", section_style))
    ratio_table_data = [
        [Paragraph("<b>Accounting Variable</b>", body_style), Paragraph("<b>Evaluated Value</b>", body_style)],
        [Paragraph("X1: Working Capital / Total Assets", body_style), Paragraph(str(data.get("x1", 0)), body_style)],
        [Paragraph("X2: Retained Earnings / Total Assets", body_style), Paragraph(str(data.get("x2", 0)), body_style)],
        [Paragraph("X3: EBITDA / Total Assets", body_style), Paragraph(str(data.get("x3", 0)), body_style)],
        [Paragraph("X4: Capitalization / Total Liabilities", body_style), Paragraph(str(data.get("x4", 0)), body_style)],
        [Paragraph("X5: Asset Turnover Speed", body_style), Paragraph(str(data.get("x5", 0)), body_style)],
        [Paragraph("Sentiment Proxy Bound", body_style), Paragraph(str(data.get("sentiment", 0)), body_style)]
    ]
    t_ratios = Table(ratio_table_data, colWidths=[340, 180])
    t_ratios.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (1, 0), colors.HexColor("#e2e8f0")),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ("PADDING", (0, 0), (-1, -1), 4)
    ]))
    story.append(t_ratios)
    
    doc.build(story)
    buffer.seek(0)
    return buffer
