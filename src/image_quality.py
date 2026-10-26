import cv2
import numpy as np


def assess_image_quality(image: np.ndarray) -> dict:
    """Return basic technical image indicators, not clinical adequacy."""
    if image is None or image.size == 0:
        raise ValueError("The uploaded image is empty or unreadable.")

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 3 and image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        raise ValueError("Unsupported image format.")

    height, width = gray.shape
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    warnings = []
    if min(height, width) < 128:
        warnings.append("Image resolution is very small.")
    if brightness < 25 or brightness > 230:
        warnings.append("Image brightness is at an extreme.")
    if contrast < 15:
        warnings.append("Image contrast is low.")
    if sharpness < 20:
        warnings.append("The image may be blurry.")

    return {
        "width": int(width),
        "height": int(height),
        "brightness": round(brightness, 2),
        "contrast": round(contrast, 2),
        "sharpness": round(sharpness, 2),
        "warnings": warnings,
    }
