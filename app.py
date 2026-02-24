# ==============================================
# FINAL PIPELINE — RETINAL ANALYSIS SYSTEM
# ==============================================

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st

os.environ["PORT"] = os.environ.get("PORT", "10000")

from retina_system.segmentation import segment_vessels
from retina_system.classification import classify_image, model
from retina_system.feature_extraction import extract_vessel_features
from retina_system.gradcam_utils import GradCAM
from retina_system.report_generator import generate_pdf_report


# -------------------------------------------------
# Device
# -------------------------------------------------
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)


# -------------------------------------------------
# OUTPUT DIRECTORY
# -------------------------------------------------
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


# -------------------------------------------------
# GRAD-CAM SETUP
# -------------------------------------------------
target_layer = model.conv_head
grad_cam = GradCAM(model, target_layer)


# -------------------------------------------------
# FULL PIPELINE FUNCTION
# -------------------------------------------------
def analyze_retina(image_path, patient_id):

    print("\n🔬 Starting Full Retinal Analysis...\n")

    image_path = Path(image_path)

    # =========================
    # 1️⃣ SEGMENTATION
    # =========================
    prob_map, final_mask = segment_vessels(image_path)

    prob_path = OUTPUT_DIR / f"{patient_id}_prob_map.png"
    mask_path = OUTPUT_DIR / f"{patient_id}_final_mask.png"

    cv2.imwrite(str(prob_path), prob_map)
    cv2.imwrite(str(mask_path), final_mask)

    print("✔ Segmentation completed")

    # =========================
    # 2️⃣ FEATURE EXTRACTION
    # =========================
    vessel_features = extract_vessel_features(final_mask)
    print("✔ Vessel features extracted")

    # =========================
    # 3️⃣ CLASSIFICATION
    # =========================
    dr_grade, confidence, probs = classify_image(image_path)
    print("✔ DR Classification completed")

    # =========================
    # 4️⃣ GRAD-CAM
    # =========================
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((300,300)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3)
    ])

    img = Image.open(image_path).convert("RGB")
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)

    class_index = np.argmax(probs)

    cam = grad_cam.generate(input_tensor, class_index)

    cam = cv2.resize(cam, (img.size[0], img.size[1]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    original = cv2.imread(str(image_path))
    overlay = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    gradcam_path = OUTPUT_DIR / f"{patient_id}_gradcam.png"
    cv2.imwrite(str(gradcam_path), overlay)

    print("✔ Grad-CAM generated")

    # =========================
    # 5️⃣ SAVE SEGMENTED IMAGE
    # =========================
    mask_resized = cv2.resize(final_mask, (original.shape[1], original.shape[0]))

    mask_color = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)

    segmented_overlay = cv2.addWeighted(
        original, 0.7,
        mask_color, 0.3,
        0
    )

    segmented_path = OUTPUT_DIR / f"{patient_id}_segmented.png"
    cv2.imwrite(str(segmented_path), segmented_overlay)

    # =========================
    # 6️⃣ GENERATE PDF REPORT
    # =========================
    report_path = OUTPUT_DIR / f"{patient_id}_clinical_report.pdf"

    generate_pdf_report(
        save_path=str(report_path),
        patient_id=patient_id,
        dr_grade=dr_grade,
        confidence=confidence,
        vessel_features=vessel_features,
        segmented_img_path=str(segmented_path),
        prob_map_path=str(prob_path),
        final_mask_path=str(mask_path),
        gradcam_path=str(gradcam_path)
    )

    print("\n🎉 FULL ANALYSIS COMPLETE")
    print("📄 Report saved at:", report_path)

    return report_path

# -------------------------------------------------
# RUN EXAMPLE
# -------------------------------------------------
# if __name__ == "__main__":
#     test_image = "APTOS/train_images/000c1434d8d7.png"
#     analyze_retina(test_image, patient_id="PAT_001")

# -------------------------------------------------
# 🛑 ROBUST FUNDUS IMAGE VALIDATION (Improved)
# -------------------------------------------------
def is_fundus_image(image_path):

    img = cv2.imread(str(image_path))
    if img is None:
        return False

    img = cv2.resize(img, (512, 512))
    h, w = img.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # -------------------------------------------------
    # 1️⃣ Check circular mask using threshold
    # -------------------------------------------------
    _, thresh = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return False

    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    circle_area = np.pi * (min(h, w)//2)**2

    circular_ratio = area / circle_area

    # Fundus images typically fill 60%+ circular region
    if circular_ratio < 0.5:
        return False

    # -------------------------------------------------
    # 2️⃣ Check dark border (fundus has black edges)
    # -------------------------------------------------
    border_pixels = np.concatenate([
        gray[0:20, :].flatten(),
        gray[-20:, :].flatten(),
        gray[:, 0:20].flatten(),
        gray[:, -20:].flatten()
    ])

    border_dark_ratio = np.mean(border_pixels < 40)

    if border_dark_ratio < 0.4:
        return False

    # -------------------------------------------------
    # 3️⃣ Check red channel dominance
    # -------------------------------------------------
    b, g, r = cv2.split(img)

    red_mean = np.mean(r)
    green_mean = np.mean(g)
    blue_mean = np.mean(b)

    if not (red_mean > green_mean and red_mean > blue_mean):
        return False

    # -------------------------------------------------
    # 4️⃣ Center brightness check
    # -------------------------------------------------
    center_crop = gray[200:312, 200:312]
    if np.mean(center_crop) < 50:
        return False

    return True

# ==============================================
# ULTRA FUTURISTIC RETINA SCANNER UI
# ==============================================

st.set_page_config(
    page_title="RetinaGrader",
    page_icon="👁️",
    layout="wide"
)

# -------------------------------------------------
# 🌌 ADVANCED RETINA SCANNER CSS
# -------------------------------------------------
st.markdown("""
<style>

/* Deep medical dark background */
.stApp {
    background: radial-gradient(circle at center, #0a0f1f 0%, #000000 70%);
    color: #E0F7FA;
}

/* Retina scan animation */
.retina-loader {
    width: 200px;
    height: 200px;
    border-radius: 50%;
    border: 4px solid rgba(0,255,255,0.3);
    border-top: 4px solid #00FFFF;
    animation: spin 2s linear infinite;
    margin: auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Glowing hospital card */
.block-container {
    background: rgba(0, 255, 255, 0.05);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 0 30px rgba(0,255,255,0.2);
}

/* Neon buttons */
.stButton>button {
    background: linear-gradient(90deg, #00ffff, #0077ff);
    color: black;
    font-weight: bold;
    border-radius: 15px;
    height: 3.2em;
    font-size: 16px;
}

/* Clinical metrics */
[data-testid="metric-container"] {
    background: rgba(0,255,255,0.08);
    border-radius: 15px;
    padding: 20px;
    box-shadow: 0 0 15px rgba(0,255,255,0.2);
}

/* Remove footer */
footer {visibility: hidden;}

</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# 👁️ FUTURISTIC HEADER
# -------------------------------------------------
st.markdown("""
<h1 style='text-align:center; font-size:60px; color:#00FFFF;'>
👁️ RETINAGRADER
</h1>
<h3 style='text-align:center; color:#00bcd4;'>
Advanced Clinical AI Retina Scanner
</h3>
""", unsafe_allow_html=True)

st.markdown("---")

# -------------------------------------------------
# 🏥 HOSPITAL SIDEBAR
# -------------------------------------------------
with st.sidebar:
    st.title("🏥 Clinical Module")
    st.markdown("""
    ### AI Diagnostic Pipeline

    ✔ Multi-Scale Vessel Segmentation  
    ✔ Vessel Feature Quantification  
    ✔ DR Severity Grading  
    ✔ Explainable AI (Grad-CAM)  
    ✔ Automated Clinical Report  

    ---
    **System Status:** 🟢 Online  
    **Inference Engine:** PyTorch  
    """)

# -------------------------------------------------
# 🏥 CENTERED MEDICAL DASHBOARD LAYOUT
# -------------------------------------------------

# Create centered container
center_col1, center_col2, center_col3 = st.columns([1, 3, 1])

with center_col2:

    st.markdown("### 📤 Upload Fundus Image")

    uploaded_file = st.file_uploader(
        "",
        type=["png","jpg","jpeg"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:

        temp_path = OUTPUT_DIR / uploaded_file.name
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.markdown("### 🖼 Retinal Scan")
        st.image(temp_path, use_container_width=True)

        st.markdown("### 🆔 Patient Information")
        patient_id = st.text_input("", value="PAT_001")

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🧠 Initiate Retina Scan", use_container_width=True):

            # -------------------------------------------------
            # 🛑 FUNDUS VALIDATION
            # -------------------------------------------------
            if not is_fundus_image(temp_path):
                st.error("🚨 Invalid Image Detected")
                st.warning("Please upload a valid retinal fundus image only.")
                st.stop()

            # -------------------------------------------------
            # 🌀 Retina Scan Loader
            # -------------------------------------------------
            loader_placeholder = st.empty()
            loader_placeholder.markdown(
                """
                <div style="text-align:center;">
                    <div class='retina-loader'></div>
                    <h3 style='color:#00FFFF;'>Scanning Retina...</h3>
                </div>
                """,
                unsafe_allow_html=True
            )

            # Run AI
            report_path = analyze_retina(temp_path, patient_id)

            loader_placeholder.empty()

            st.success("✅ Retina Scan Completed")

            # -------------------------------------------------
            # 🧬 CLINICAL DIAGNOSTIC PANEL
            # -------------------------------------------------
            st.markdown("---")
            st.markdown("## 🧬 Clinical Diagnostic Panel")

            dr_grade, confidence, _ = classify_image(temp_path)

            # ✅ FIXED RISK LOGIC
            grade = dr_grade.lower().strip()

            if "proliferative" in grade:
                risk = "Critical 🚨"
            elif "severe" in grade:
                risk = "High ⚠️"
            elif "moderate" in grade:
                risk = "Moderate ⚡"
            elif "mild" in grade:
                risk = "Low"
            else:
                risk = "No Risk"

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            metric_col1.metric("DR Grade", dr_grade)
            metric_col2.metric("Confidence", f"{confidence*100:.2f}%")
            metric_col3.metric("Risk Level", risk)

            st.markdown("---")

            # -------------------------------------------------
            # 📄 Download Report
            # -------------------------------------------------
            with open(report_path, "rb") as f:
                st.download_button(
                    "📄 Download Clinical Report",
                    data=f,
                    file_name=f"{patient_id}_clinical_report.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
