import io
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

def generate_pdf_report(data: dict) -> io.BytesIO:
    """
    Dynamically generates a clean, institutional-grade credit risk report in memory.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=40, leftMargin=40, topMargin=40, bottomMargin=40
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette Styling
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=24,
        leading=28,
        textColor=colors.HexColor("#0f172a"), # Slate 900
        spaceAfter=6
    )
    
    section_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=14,
        leading=18,
        textColor=colors.HexColor("#0284c7"), # Sky 600
        spaceBefore=14,
        spaceAfter=8,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#334155") # Slate 700
    )

    story = []
    
    # 1. Header Band
    story.append(Paragraph("EXECUTIVE CREDIT RISK EVALUATION REPORT", title_style))
    story.append(Paragraph(f"Target Entity: <b>{data.get('company_name', 'Unknown Enterprise')}</b>", body_style))
    story.append(Spacer(1, 15))
    
    # 2. Risk Metrics Dashboard Grid
    story.append(Paragraph("1. Core Predictive Risk Analytics", section_style))
    
    # Define color mappings for the PDF based on the risk zone
    zone = data.get('financial_risk_zone', 'Grey Zone')
    zone_color = "#10b981" if "Safe" in zone else ("#f59e0b" if "Grey" in zone else "#ef4444")
    
    summary_table_data = [
        [
            Paragraph("<b>Altman Z-Score Matrix Metric</b>", body_style),
            Paragraph(f"<font color='{zone_color}'><b>{data.get('altman_z_score', '0.0')} ({zone})</b></font>", body_style)
        ],
        [
            Paragraph("<b>XGBoost ML Default Probability</b>", body_style),
            Paragraph(f"<b>{data.get('ml_risk_probability', '0.0')}% Probability</b>", body_style)
        ],
        [
            Paragraph("<b>Model Classification Decision</b>", body_style),
            Paragraph(f"<b>{data.get('model_classification', 'No Risk')}</b>", body_style)
        ]
    ]
    
    t_summary = Table(summary_table_data, colWidths=[240, 280])
    t_summary.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
    ]))
    story.append(t_summary)
    story.append(Spacer(1, 15))
    
    # 3. Input Feature Breakdown
    story.append(Paragraph("2. Extracted Financial Ratio Vectors", section_style))
    
    ratio_table_data = [
        [Paragraph("<b>Accounting Variable Vector Map</b>", body_style), Paragraph("<b>Evaluated Weight Ratio Value</b>", body_style)],
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
        ('TOPPADDING', (0, 0), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(t_ratios)
    story.append(Spacer(1, 20))
    
    # 4. Standard Regulatory/Institutional Legal Disclaimer Band
    story.append(Paragraph("<b>Notice & Evaluation Disclaimer:</b>", ParagraphStyle('Notice', parent=body_style, fontSize=8, leading=10, fontName="Helvetica-Bold")))
    story.append(Paragraph("This algorithmic assessment report is derived via a synthetic dual-model engine integrating traditional Altman Z-Score modeling configurations with gradient-boosted tree decision matrices. This document is compiled for academic and technical application presentation scenarios only. It does not map into official financial or legal fiduciary investment directions.", ParagraphStyle('DiscText', parent=body_style, fontSize=8, leading=11, textColor=colors.HexColor("#64748b"))))
    
    doc.build(story)
    buffer.seek(0)
    return buffer