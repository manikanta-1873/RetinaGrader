# 🩺 RetinaGrader  
### AI-Powered Retinal Analysis & Diabetic Retinopathy Detection System

RetinaGrader is an end-to-end AI-based clinical decision support system for automated retinal vessel segmentation, diabetic retinopathy (DR) grading, explainability visualization, and structured clinical report generation.

The system is deployed using **Streamlit** and provides real-time retinal analysis with AI-generated clinical reports.

---

## 🚀 Live Deployment

🌐 Deployed via Streamlit  
Upload a retinal fundus image and receive:

- Vessel segmentation
- DR severity classification
- Grad-CAM explainability
- Clinical metrics
- Downloadable PDF report

---

# 📌 Features

## 🔬 1. Retinal Vessel Segmentation (MSF-Net)

- Multi-Scale Fusion Network (MSF-Net)
- Probability map generation
- Vessel refinement using CLAHE + Adaptive Thresholding
- Final clean vessel mask
- Vessel overlay visualization

---

## 📊 2. Vessel Feature Extraction

From the segmented vessel mask:

- Vessel Density (%)
- Tortuosity Index
- Branch Point Count

These features enhance interpretability and support severity analysis.

---

## 🧠 3. Diabetic Retinopathy Classification

- EfficientNet-B3 (via `timm`)
- 5-Class DR grading:
  - No DR
  - Mild
  - Moderate
  - Severe
  - Proliferative DR
- Confidence score output

---

## 🔥 4. Explainability (Grad-CAM)

- Heatmap visualization of diagnostic focus regions
- Identifies high-impact retinal areas
- Improves clinical trust and transparency

---

## 📄 5. Automated Clinical Report Generation

Generates structured PDF report including:

- Patient ID
- DR Grade
- Confidence Score
- Risk Level
- Clinical Recommendation
- Vessel Metrics
- Segmented Vessel Image
- Probability Map
- Final Mask
- Grad-CAM Visualization
- AI Disclaimer

---

# 🏗️ System Architecture

```
User Upload (Streamlit UI)
        ↓
Image Preprocessing
        ↓
MSF-Net Vessel Segmentation
        ↓
Probability Map → Refinement → Final Mask
        ↓
Feature Extraction (Density, Tortuosity, Branch Points)
        ↓
EfficientNet-B3 DR Classification
        ↓
Grad-CAM Explainability
        ↓
Risk Assessment Logic
        ↓
PDF Clinical Report Generation
```

---

# 📂 Project Structure

```
RetinaGrader/
│
├── app.py                      # Streamlit entry point
├── requirements.txt
│
├── retina_system/
│   ├── segmentation.py         # MSF-Net inference + vessel refinement
│   ├── classification.py       # EfficientNet-B3 + Grad-CAM
│   ├── report_generator.py     # Clinical PDF report
│
├── msfnet_model.py
├── efficientnet_b3_best.pth
├── msfnet_best.pth
│
├── outputs/                    # Generated masks & reports
└── README.md
```

---

# 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/manikanta-1873/RetinaGrader.git
cd RetinaGrader
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Run Locally (Streamlit)

```bash
streamlit run app.py
```

Then open:

```
http://localhost:8501
```

---

# 🧪 Model Details

## Vessel Segmentation
- Custom MSF-Net architecture
- Trained on retinal vessel datasets
- Output: Probability map + binary mask

## DR Classification
- EfficientNet-B3
- Transfer learning
- Weighted cross-entropy
- CosineAnnealingLR scheduler
- Class imbalance handling

---

# 📈 Performance Metrics

- Validation Accuracy: ~83%
- Multi-class ROC-AUC: ~0.93
- Cohen’s Kappa: ~0.75
- Explainability supported via Grad-CAM

---

# 🔍 Explainability & Responsible AI

RetinaGrader integrates:

- Pixel-level segmentation
- Feature-level metrics
- Grad-CAM heatmaps
- Risk level logic
- Structured medical disclaimer

Designed as a clinical decision-support tool, not a replacement for medical diagnosis.

---

# ⚠️ Clinical Disclaimer

This system is intended for clinical decision-support purposes only.  
Final diagnosis and treatment decisions must be made by a licensed and qualified ophthalmologist.

---

# 🚀 Future Enhancements

- Multi-task learning (joint segmentation + classification)
- Lesion detection module
- Risk progression prediction
- Model uncertainty estimation
- Cross-dataset validation
- Cloud deployment (AWS / GCP)

---

# ⭐ If You Found This Useful

Give this repository a ⭐ and support the project.

---

**RetinaGrader — AI-Assisted Retinal Intelligence**
