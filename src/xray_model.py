
"""Research-only wrapper for TorchXRayVision's pretrained chest X-ray model.

Model outputs are not calibrated patient-specific disease probabilities.
This application is for research and education, not clinical diagnosis.
"""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _load_model():
    try:
        import torch
        import torchxrayvision as xrv
    except ImportError as exc:
        raise RuntimeError(
            "Model dependencies are missing. Run: "
            "python -m pip install -r requirements.txt"
        ) from exc

    # The first run may download pretrained weights.
    model = xrv.models.DenseNet(weights="densenet121-res224-all")
    model.eval()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    return model, torch, xrv, device


def get_model_info() -> dict:
    """Return model details for the application interface."""
    model, _, _, device = _load_model()

    return {
        "model": "TorchXRayVision DenseNet-121",
        "weights": "densenet121-res224-all",
        "device": device,
        "labels": list(model.pathologies),
    }


def predict_findings(rgb_image: np.ndarray) -> dict:
    """Return model scores for the supported pathology labels.

    Args:
        rgb_image: RGB image represented as a NumPy array.

    Returns:
        Dictionary mapping each model label to its output score.
    """
    if rgb_image is None or rgb_image.size == 0:
        raise ValueError("The uploaded image is empty.")

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels.")

    if rgb_image.dtype != np.uint8:
        raise ValueError("Expected an RGB image with uint8 pixel values.")

    model, torch, xrv, device = _load_model()

    import cv2

    # Convert RGB to grayscale.
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)

    # Apply TorchXRayVision's intensity normalization.
    image = xrv.datasets.normalize(gray, 255.0)

    # TorchXRayVision transforms expect channel-first image data: (C, H, W).
    image = image[np.newaxis, :, :]

    # Use the model's expected center-crop and resize transforms.
    image = xrv.datasets.XRayCenterCrop()(image)
    image = xrv.datasets.XRayResizer(224)(image)

    # Add the batch dimension: (1, C, H, W).
    image_tensor = torch.from_numpy(
        np.asarray(image, dtype=np.float32)
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image_tensor)[0].detach().cpu().numpy()

    labels = list(model.pathologies)

    if len(labels) != len(output):
        raise RuntimeError(
            "Model output count does not match the number of labels."
        )

    if not np.isfinite(output).all():
        raise RuntimeError("The model returned invalid numerical outputs.")

    return {
        str(label): float(score)
        for label, score in zip(labels, output)
    }
