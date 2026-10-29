
"""Coordinate the XrayGuide AI analysis workflow."""

import cv2
import numpy as np

from src.agent_controller import decide_next_action
from src.opencv_processor import enhance_contrast
from src.xray_model import predict_findings
from src.explanation import generate_explanation
from src.gradcam import generate_gradcam


def analyze_xray(rgb_image: np.ndarray) -> dict:
    """
    Coordinate image validation, OpenCV enhancement, model inference,
    explanation generation and optional Grad-CAM visualization.

    Research prototype only; not a medical decision system.
    """

    # ========================================================
    # 1. Validate the uploaded image
    # ========================================================

    if (
        rgb_image is None
        or not isinstance(rgb_image, np.ndarray)
        or rgb_image.size == 0
    ):
        return {
            "status": "error",
            "message": "Please upload a valid image.",
            "decision": {
                "action": "reupload",
                "status": "error",
                "reason": "The uploaded image is empty or invalid.",
            },
        }

    if rgb_image.ndim != 3 or rgb_image.shape[2] != 3:
        return {
            "status": "error",
            "message": "Expected an RGB image with three channels.",
            "decision": {
                "action": "reupload",
                "status": "error",
                "reason": (
                    "The image must have three RGB colour channels."
                ),
            },
        }

    if rgb_image.dtype != np.uint8:
        return {
            "status": "error",
            "message": "Expected an 8-bit RGB image.",
            "decision": {
                "action": "reupload",
                "status": "error",
                "reason": (
                    "The image must use the uint8 data type."
                ),
            },
        }

    height, width = rgb_image.shape[:2]

    metrics = {
        "width": int(width),
        "height": int(height),
        "warnings": [],
    }

    # Ask the controller whether the image can proceed.
    decision = decide_next_action(metrics)

    if decision["action"] != "run_model":
        return {
            "status": "error",
            "message": decision["reason"],
            "decision": decision,
        }

    # ========================================================
    # 2. OpenCV enhancement
    # ========================================================

    try:
        bgr_image = cv2.cvtColor(
            rgb_image,
            cv2.COLOR_RGB2BGR,
        )

        enhanced_gray = enhance_contrast(bgr_image)

        if enhanced_gray is None or np.asarray(enhanced_gray).size == 0:
            raise ValueError(
                "OpenCV enhancement returned an empty image."
            )

        enhanced_gray = np.asarray(enhanced_gray)

    except Exception as exc:
        decision = decide_next_action(
            metrics,
            error=f"OpenCV enhancement failed: {exc}",
        )

        return {
            "status": "error",
            "message": decision["reason"],
            "decision": decision,
        }

    # ========================================================
    # 3. Run the pretrained vision model
    # ========================================================

    try:
        # Preserve the model's own preprocessing pipeline.
        # Predictions use the original RGB image.
        predictions = predict_findings(rgb_image)

    except Exception as exc:
        decision = decide_next_action(
            metrics,
            error=f"Model inference failed: {exc}",
        )

        return {
            "status": "error",
            "message": decision["reason"],
            "enhanced_image": enhanced_gray,
            "decision": decision,
        }

    decision = decide_next_action(
        metrics,
        predictions=predictions,
    )

    if decision["action"] != "generate_explanation":
        return {
            "status": "error",
            "message": decision["reason"],
            "enhanced_image": enhanced_gray,
            "predictions": predictions,
            "decision": decision,
        }

    # ========================================================
    # 4. Generate the text explanation
    # ========================================================

    try:
        explanation = generate_explanation(
            predictions,
            top_n=1,
        )

    except Exception as exc:
        decision = decide_next_action(
            metrics,
            predictions=predictions,
            error=f"Explanation generation failed: {exc}",
        )

        return {
            "status": "error",
            "message": decision["reason"],
            "enhanced_image": enhanced_gray,
            "predictions": predictions,
            "decision": decision,
        }

    decision = decide_next_action(
        metrics,
        predictions=predictions,
        explanation=explanation,
    )

    if decision["action"] != "generate_gradcam":
        return {
            "status": "error",
            "message": decision["reason"],
            "enhanced_image": enhanced_gray,
            "predictions": predictions,
            "explanation": explanation,
            "decision": decision,
        }

    # ========================================================
    # 5. Generate optional Grad-CAM
    # ========================================================

    top_label = max(predictions, key=predictions.get)

    gradcam = None
    gradcam_error = None

    try:
        gradcam = generate_gradcam(
            rgb_image,
            top_label,
        )

        # Treat missing or malformed Grad-CAM output as unavailable.
        if (
            not isinstance(gradcam, dict)
            or gradcam.get("heatmap") is None
        ):
            gradcam_error = (
                "Grad-CAM returned no usable heatmap."
            )
            gradcam = None

    except Exception as exc:
        gradcam_error = str(exc)
        gradcam = None

    # ========================================================
    # 6. Let the controller choose the final action
    # ========================================================

    decision = decide_next_action(
        metrics,
        predictions=predictions,
        explanation=explanation,
        gradcam=gradcam if gradcam is not None else {"heatmap": None},
    )

    # A missing Grad-CAM must not discard valid model outputs.
    if decision["action"] == "continue_without_gradcam":
        return {
            "status": "success",
            "original_image": rgb_image,
            "enhanced_image": enhanced_gray,
            "predictions": predictions,
            "top_label": top_label,
            "explanation": explanation,
            "gradcam": None,
            "gradcam_error": (
                gradcam_error
                or "Grad-CAM is unavailable for this image."
            ),
            "decision": decision,
        }

    if decision["action"] != "present_results":
        return {
            "status": "error",
            "message": decision["reason"],
            "enhanced_image": enhanced_gray,
            "predictions": predictions,
            "explanation": explanation,
            "gradcam": gradcam,
            "gradcam_error": gradcam_error,
            "decision": decision,
        }

    # ========================================================
    # 7. Return all results to Streamlit
    # ========================================================

    return {
        "status": "success",
        "original_image": rgb_image,
        "enhanced_image": enhanced_gray,
        "predictions": predictions,
        "top_label": top_label,
        "decision": decision,
        "explanation": explanation,
        "gradcam": gradcam,
        "gradcam_error": None,
    }
