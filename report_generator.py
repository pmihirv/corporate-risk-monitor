import os
import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(data: dict) -> io.BytesIO:
    """
    Dynamically generates an advanced, institutional-grade credit risk report 
    with embedded trailing ratio lines and SHAP feature attribution plots.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=22,
        leading=26,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0284c7"),
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155")
    )

    story = []
    
    # 1. Document Header Block
    company_name = data.get('company_name', 'Unknown Enterprise')
    story.append(Paragraph("EXECUTIVE CREDIT RISK EVALUATION REPORT", title_style))
    story.append(Paragraph(f"Target Entity Asset Profile: <b>{company_name}</b>", body_style))
    story.append(Spacer(1, 10))
    
    # 2. Predictive Risk Analytics Scorecard
    story.append(Paragraph("1. Core Predictive Risk Analytics", section_style))
    zone = data.get('financial_risk_zone', 'Grey Zone')
    zone_color = "#10b981" if "Safe" in zone else ("#f59e0b" if "Grey" in zone else "#ef4444")
    
    summary_table_data = [
        [Paragraph("<b>Altman Z-Score Vector Matrix</b>", body_style), 
         Paragraph(f"<font color='{zone_color}'><b>{data.get('altman_z_score', '0.0')} ({zone})</b></font>", body_style)],
        [Paragraph("<b>XGBoost ML Default Probability</b>", body_style), 
         Paragraph(f"<b>{data.get('ml_risk_probability', '0.0')}% Probability</b>", body_style)],
        [Paragraph("<b>Model Classification Decision</b>", body_style), 
         Paragraph(f"<b>{data.get('model_classification', 'No Risk')}</b>", body_style)]
    ]
    
    t_summary = Table(summary_table_data, colWidths=[240, 280])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 12))
    
    # 3. Dynamic Visual Research Panels Injection Layer
    story.append(Paragraph("2. Deep-Dive Explainability & Trajectory Analysis Charts", section_style))
    
    # Extract structural clean string ticker token to locate visual PNG paths
    ticker_token = str(data.get('ticker_symbol', 'CUSTOM')).strip().upper()
    visuals_dir = "data/visuals"
    
    trend_img_path = os.path.join(visuals_dir, f"{ticker_token}_trends.png")
    shap_img_path = os.path.join(visuals_dir, f"{ticker_token}_shap.png")
    
    # Construct a side-by-side table layout if visual assets are generated on disk
    visual_cells = []
    if os.path.exists(trend_img_path):
        visual_cells.append(Image(trend_img_path, width=250, height=142))
    else:
        visual_cells.append(Paragraph("<i>Trajectory chart not compiled for this asset. Run visualize.py to seed.</i>", body_style))
        
    if os.path.exists(shap_img_path):
        visual_cells.append(Image(shap_img_path, width=250, height=142))
    else:
        visual_cells.append(Paragraph("<i>SHAP risk contribution chart not compiled for this asset.</i>", body_style))
        
    t_visuals = Table([visual_cells], colWidths=[260, 260])
    t_visuals.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
    ]))
    story.append(t_visuals)
    story.append(Spacer(1, 12))
    
    # 4. Input Feature Table
    story.append(Paragraph("3. Extracted Financial Ratio Vectors", section_style))
    ratio_table_data = [
        [Paragraph("<b>Accounting Variable Vector Map</b>", body_style), Paragraph("<b>Evaluated Ratio Value</b>", body_style)],
        [Paragraph("X1: Working Capital / Total Assets", body_style), Paragraph(str(data.get('x1', 0)), body_style)],
        [Paragraph("X2: Retained Earnings / Total Assets", body_style), Paragraph(str(data.get('x2', 0)), body_style)],
        [Paragraph("X3: EBITDA / Total Assets", body_style), Paragraph(str(data.get('x3', 0)), body_style)],
        [Paragraph("X4: Equity Capitalization / Total Liabilities", body_style), Paragraph(str(data.get('x4', 0)), body_style)],
        [Paragraph("X5: Asset Turnover Speed (Sales / Total Assets)", body_style), Paragraph(str(data.get('x5', 0)), body_style)],
        [Paragraph("Textual Sentiment Density (FinBERT Proxy Bound)", body_style), Paragraph(str(data.get('sentiment', 0)), body_style)]
    ]
    
    t_ratios = Table(ratio_table_data, colWidths=[340, 180])
    t_ratios.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (1, 0), colors.HexColor("#e2e8f0")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_ratios)
    story.append(Spacer(1, 14))
    
    # 5. Regulatory Notice Disclaimer Panel
    story.append(Paragraph("<b>Notice & Evaluation Disclaimer:</b>", ParagraphStyle('Notice', parent=body_style, fontSize=7, fontName="Helvetica-Bold")))
    story.append(Paragraph("This algorithmic assessment report is derived via a production-grade machine learning model trained on real-world US historical default matrices across out-of-sample corporate profiles. Localized feature attributions are parsed using serialized SHAP mathematical tree kernels. This document is compiled strictly for personal research exploration and software testing scenarios.", ParagraphStyle('DiscText', parent=body_style, fontSize=7, leading=9, textColor=colors.HexColor("#64748b"))))
    
    doc.build(story)
    buffer.seek(0)
    return buffer