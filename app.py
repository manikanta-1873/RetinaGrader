# ==============================================
# FINAL PIPELINE — RETINAL ANALYSIS SYSTEM
# ==============================================

import os
import cv2
import torch
import numpy as np
from pathlib import Path
from PIL import Image

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
if __name__ == "__main__":
    test_image = "APTOS/train_images/000c1434d8d7.png"
    analyze_retina(test_image, patient_id="PAT_001")