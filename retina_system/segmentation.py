# retina_system/segmentation.py

import torch
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
from pathlib import Path

from msfnet_model import MSFNet  # your existing file

# -----------------------------
# Device
# -----------------------------
DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

# -----------------------------
# Load MSF-Net
# -----------------------------
MODEL_PATH = Path("models/msfnet_best.pth")

msf_model = MSFNet()
msf_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
msf_model.to(DEVICE)
msf_model.eval()

# -----------------------------
# Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# -----------------------------
# Refinement
# -----------------------------
def refine_vessels(prob_img):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(prob_img)

    blurred = cv2.GaussianBlur(enhanced, (5,5), 0)

    binary = cv2.adaptiveThreshold(
        blurred,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        15,
        -5
    )

    kernel = np.ones((3,3), np.uint8)
    clean = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    clean = cv2.morphologyEx(clean, cv2.MORPH_CLOSE, kernel)

    return clean


# -----------------------------
# Main Segmentation Function
# -----------------------------
def segment_vessels(image_path):
    """
    Returns:
        prob_map (0–255 uint8)
        final_mask (binary uint8)
    """

    image = Image.open(image_path).convert("L")  # grayscale
    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = msf_model(input_tensor)
        prob = torch.sigmoid(output)

    prob_map = prob[0,0].cpu().numpy()
    prob_map_uint8 = (prob_map * 255).astype(np.uint8)

    final_mask = refine_vessels(prob_map_uint8)

    return prob_map_uint8, final_mask