from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from datetime import datetime


def create_prediction_report(filepath, patient_name, username, disease, confidence, image_name, top3):
    """Generate a PDF report for skin disease prediction"""
    
    doc = SimpleDocTemplate(filepath, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=1,  # Center
        textColor=colors.HexColor('#0066cc')
    )
    story.append(Paragraph("Skin Disease Detection Report", title_style))
    story.append(Spacer(1, 12))
    
    # Header info
    header_style = ParagraphStyle(
        'HeaderStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.grey
    )
    story.append(Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", header_style))
    story.append(Spacer(1, 12))
    
    # Patient Information
    story.append(Paragraph("Patient Information", styles['Heading2']))
    info_data = [
        ["Patient Name:", patient_name],
        ["Username:", username],
        ["Image:", image_name]
    ]
    info_table = Table(info_data, colWidths=[2*inch, 3*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 20))
    
    # Prediction Results
    story.append(Paragraph("Prediction Results", styles['Heading2']))
    
    # Main prediction
    result_style = ParagraphStyle(
        'ResultStyle',
        parent=styles['Normal'],
        fontSize=16,
        textColor=colors.HexColor('#0066cc'),
        spaceAfter=10
    )
    story.append(Paragraph(f"<b>Predicted Disease:</b> {disease}", result_style))
    story.append(Paragraph(f"<b>Confidence:</b> {confidence:.2f}%", styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Top 3 predictions
    story.append(Paragraph("Top Predictions", styles['Heading3']))
    top_data = [["Rank", "Disease", "Confidence"]]
    for rank, (label, score) in enumerate(top3, start=1):
        top_data.append([str(rank), label, f"{score:.2f}%"])
    
    top_table = Table(top_data, colWidths=[1*inch, 3*inch, 2*inch])
    top_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0066cc')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(top_table)
    story.append(Spacer(1, 20))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'DisclaimerStyle',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.red,
        alignment=1
    )
    story.append(Paragraph(
        "MEDICAL DISCLAIMER: This report is for educational and research purposes only. "
        "It is not a substitute for professional medical diagnosis or treatment. "
        "Please consult a qualified healthcare professional.",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(story)