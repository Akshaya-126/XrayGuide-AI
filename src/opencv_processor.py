import cv2
import numpy as np


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to a grayscale representation for visual comparison only."""
    if image is None or image.size == 0:
        raise ValueError("Cannot process an empty image.")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image format.")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)
