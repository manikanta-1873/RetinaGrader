# retina_system/report_generator.py

import os
from datetime import datetime
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image, Table, TableStyle, KeepTogether
)
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER


# ---------------------------------------------------------
# Risk Logic
# ---------------------------------------------------------
def risk_level(dr_grade):
    mapping = {
        "No DR": "Low Risk",
        "Mild": "Mild Risk",
        "Moderate": "Moderate Risk",
        "Severe": "High Risk",
        "Proliferative DR": "Critical Risk"
    }
    return mapping.get(dr_grade, "Undetermined")


def recommendation(dr_grade):
    mapping = {
        "No DR": "Annual retinal screening recommended.",
        "Mild": "Follow-up examination in 6–12 months.",
        "Moderate": "Consult ophthalmologist within 3 months.",
        "Severe": "Immediate specialist referral required.",
        "Proliferative DR": "Urgent treatment required."
    }
    return mapping.get(dr_grade, "Further clinical evaluation recommended.")


# ---------------------------------------------------------
# Interpretation Paragraph
# ---------------------------------------------------------
def interpretation_text(dr_grade, confidence):
    return (
        f"The AI-based deep learning diagnostic system predicts "
        f"<b>{dr_grade}</b> with a confidence score of "
        f"<b>{confidence*100:.2f}%</b>. "
        f"Grad-CAM visualization highlights clinically relevant "
        f"retinal regions influencing the final decision."
    )


# ---------------------------------------------------------
# Helper: Add Image Block (No Page Break Issue)
# ---------------------------------------------------------
def image_block(title, image_path, styles):
    centered_style = ParagraphStyle(
        name="CenterCaption",
        parent=styles["Normal"],
        alignment=TA_CENTER
    )

    img = Image(image_path, width=4.8*inch, height=4.8*inch)
    caption = Paragraph(f"<b>{title}</b>", centered_style)

    return KeepTogether([
        caption,
        Spacer(1, 0.15 * inch),
        img,
        Spacer(1, 0.4 * inch)
    ])


# ---------------------------------------------------------
# Main Report Generator
# ---------------------------------------------------------
def generate_pdf_report(
    save_path,
    patient_id,
    dr_grade,
    confidence,
    vessel_features,
    segmented_img_path,
    prob_map_path,
    final_mask_path,
    gradcam_path
):

    doc = SimpleDocTemplate(save_path)
    elements = []
    styles = getSampleStyleSheet()

    # =====================================================
    # HEADER
    # =====================================================
    elements.append(Paragraph("<b>Advanced Clinical Vision (ACV)</b>", styles["Heading1"]))
    elements.append(Paragraph("AI-Assisted Retinal Diagnostic Report", styles["Title"]))
    elements.append(Spacer(1, 0.2 * inch))

    report_id = f"ACV-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    date_str = datetime.now().strftime("%d %B %Y, %H:%M")

    header_data = [
        ["Report ID:", report_id],
        ["Patient ID:", patient_id],
        ["Date:", date_str],
    ]

    header_table = Table(header_data, colWidths=[1.5*inch, 4.5*inch])
    header_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,-1), colors.whitesmoke)
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 0.5 * inch))

    # =====================================================
    # SECTION I — VESSEL ANALYSIS
    # =====================================================
    elements.append(Paragraph("<b>I. Automated Retinal Vessel Analysis</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))

    vessel_data = [
        ["Vessel Density:", f"{vessel_features['Vessel Density']*100:.2f}%"],
        ["Tortuosity Index:", vessel_features["Tortuosity Index"]],
        ["Branch Points:", vessel_features["Branch Points"]],
    ]

    vessel_table = Table(vessel_data, colWidths=[2*inch, 4*inch])
    vessel_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
    ]))

    elements.append(vessel_table)
    elements.append(Spacer(1, 0.5 * inch))

    # =====================================================
    # SECTION II — DR ASSESSMENT
    # =====================================================
    elements.append(Paragraph("<b>II. Diabetic Retinopathy Assessment</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))

    dr_data = [
        ["Predicted DR Grade:", dr_grade],
        ["Confidence Score:", f"{confidence*100:.2f}%"],
        ["Risk Level:", risk_level(dr_grade)],
        ["Clinical Recommendation:", recommendation(dr_grade)]
    ]

    dr_table = Table(dr_data, colWidths=[2*inch, 4*inch])
    dr_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.whitesmoke)
    ]))

    elements.append(dr_table)
    elements.append(Spacer(1, 0.4 * inch))

    elements.append(Paragraph(interpretation_text(dr_grade, confidence), styles["Normal"]))
    elements.append(Spacer(1, 0.6 * inch))

    # =====================================================
    # SECTION III — VISUAL DIAGNOSTIC OUTPUTS
    # =====================================================
    elements.append(Paragraph("<b>III. Visual Diagnostic Outputs</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.4 * inch))

    elements.append(image_block("Segmented Vessel Image", segmented_img_path, styles))
    elements.append(image_block("Probability Map", prob_map_path, styles))
    elements.append(image_block("Final Vessel Mask", final_mask_path, styles))
    elements.append(image_block("Grad-CAM Explainability Map", gradcam_path, styles))

    elements.append(Spacer(1, 0.3 * inch))

    centered_style = ParagraphStyle(
        name="CenterNote",
        parent=styles["Italic"],
        alignment=TA_CENTER
    )

    elements.append(Paragraph(
        "Red regions indicate areas of high diagnostic importance. "
        "Blue regions indicate lower contribution to the model decision.",
        centered_style
    ))

    elements.append(Spacer(1, 0.6 * inch))

    # =====================================================
    # DISCLAIMER
    # =====================================================
    elements.append(Paragraph("<b>IV. Clinical Disclaimer</b>", styles["Heading2"]))
    elements.append(Spacer(1, 0.3 * inch))

    elements.append(Paragraph(
        "This report is generated using an automated AI-based diagnostic system "
        "and is intended for clinical decision-support purposes only. "
        "Final diagnosis and treatment decisions must be made by a licensed ophthalmologist.",
        styles["Normal"]
    ))

    doc.build(elements)

    print("✅ Fully Structured Clinical-Grade Report Generated:", save_path)