import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import convolve

def extract_vessel_features(mask):
    binary = mask > 0

    vessel_density = np.sum(binary) / binary.size

    skeleton = skeletonize(binary)

    # Branch points
    kernel = np.array([[1,1,1],
                       [1,10,1],
                       [1,1,1]])

    neighbors = convolve(skeleton.astype(int), kernel)
    branch_points = np.sum(neighbors >= 13)

    tortuosity = np.sum(skeleton) / (np.sum(binary) + 1e-8)

    return {
        "Vessel Density": round(float(vessel_density), 4),
        "Tortuosity Index": round(float(tortuosity), 4),
        "Branch Points": int(branch_points)
    }