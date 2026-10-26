
"""Experimental Grad-CAM visualization for XrayGuide AI.

The heatmap visualizes regions influencing one model output.
It is not a disease-localization map or a clinical explanation.
"""

import numpy as np


def _preprocess_image(rgb_image, torch, xrv, device):
    """Prepare an RGB image using the model's preprocessing pipeline."""
    import cv2

    if rgb_image is None or rgb_image.size == 0:
        raise ValueError("The uploaded image is empty.")

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        raise ValueError("Expected an RGB image with three channels.")

    if rgb_image.dtype != np.uint8:
        raise ValueError("Expected an RGB image with uint8 pixel values.")

    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    gray = gray.astype(np.float32)

    image = xrv.datasets.normalize(gray, 255.0)
    image = image[np.newaxis, :, :]
    image = xrv.datasets.XRayCenterCrop()(image)
    image = xrv.datasets.XRayResizer(224)(image)

    tensor = torch.from_numpy(
        np.asarray(image, dtype=np.float32)
    ).unsqueeze(0).to(device)

    return tensor


def _find_target_layer(model):
    """Find the final DenseNet feature normalization layer."""
    modules = dict(model.named_modules())

    candidates = (
        "features.norm5",
        "model.features.norm5",
        "densenet.features.norm5",
    )

    for name in candidates:
        if name in modules:
            return name, modules[name]

    # Accommodate wrappers whose DenseNet is nested differently.
    matches = [
        (name, module)
        for name, module in modules.items()
        if name.endswith("features.norm5")
    ]

    if matches:
        return matches[-1]

    raise RuntimeError(
        "Could not locate DenseNet's final feature layer for Grad-CAM. "
        "Check the installed TorchXRayVision model architecture."
    )


def generate_gradcam(
    rgb_image: np.ndarray,
    target_label: str,
) -> dict:
    """Generate a Grad-CAM heatmap for one supported model label.

    Returns:
        {
            "heatmap": 2D float32 array normalized to [0, 1],
            "label": selected label,
            "target_layer": layer used,
            "raw_score": selected model output
        }
    """
    # Reuse the same cached model as normal inference.
    from src.xray_model import _load_model

    model, torch, xrv, device = _load_model()

    labels = [str(label) for label in model.pathologies]

    if target_label not in labels:
        raise ValueError(
            f"Unsupported label: {target_label}. "
            "Select a label returned by the pretrained model."
        )

    target_index = labels.index(target_label)
    target_layer_name, target_layer = _find_target_layer(model)

    input_tensor = _preprocess_image(
        rgb_image, torch, xrv, device
    )

    captured = {
        "activations": None,
        "gradients": None,
    }

    def forward_hook(_module, _inputs, output):
        if not isinstance(output, torch.Tensor):
            raise RuntimeError(
                "The selected Grad-CAM layer did not return a tensor."
            )

        captured["activations"] = output

        if output.requires_grad:
            output.register_hook(
                lambda gradient: captured.__setitem__(
                    "gradients", gradient
                )
            )

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.eval()
        model.zero_grad()

        # Gradients must be enabled for Grad-CAM.
        with torch.enable_grad():
            output = model(input_tensor)

            if output.ndim != 2 or output.shape[0] != 1:
                raise RuntimeError(
                    "Unexpected model output shape for Grad-CAM: "
                    f"{tuple(output.shape)}"
                )

            if output.shape[1] != len(labels):
                raise RuntimeError(
                    "Model output count does not match its label list."
                )

            score = output[0, target_index]
            raw_score = float(score.detach().cpu().item())

            score.backward()

        activations = captured["activations"]
        gradients = captured["gradients"]

        if activations is None or gradients is None:
            raise RuntimeError(
                "Grad-CAM could not capture feature activations "
                "and gradients from the selected layer."
            )

        # Average gradients over the spatial dimensions.
        weights = gradients.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = torch.relu(
            (weights * activations).sum(dim=1, keepdim=True)
        )

        cam = torch.nn.functional.interpolate(
            cam,
            size=(rgb_image.shape[0], rgb_image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        heatmap = cam[0, 0].detach().cpu().numpy()
        heatmap = np.maximum(heatmap, 0)

        if not np.isfinite(heatmap).all():
            raise RuntimeError(
                "Grad-CAM generated invalid numerical values."
            )

        maximum = float(heatmap.max())

        if maximum > 0:
            heatmap = heatmap / maximum
        else:
            raise RuntimeError(
                "The generated heatmap contains no positive activations."
            )

        return {
            "heatmap": heatmap.astype(np.float32),
            "label": target_label,
            "target_layer": target_layer_name,
            "raw_score": raw_score,
        }

    finally:
        handle.remove()
        model.zero_grad()
