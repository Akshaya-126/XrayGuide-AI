
"""Rule-based workflow controller for XrayGuide AI."""


def decide_next_action(
    metrics: dict,
    predictions: dict | None = None,
    error: str | None = None,
    explanation: dict | None = None,
    gradcam: dict | None = None,
) -> dict:
    """
    Select the next workflow action.

    This transparent, rule-based controller coordinates workflow stages.
    It is not an AI model or a medical decision system.
    """

    # 1. Handle processing errors.
    if error:
        return {
            "action": "stop",
            "status": "error",
            "reason": f"Workflow step failed: {error}",
        }

    # 2. Validate image metrics.
    if not isinstance(metrics, dict):
        return {
            "action": "reupload",
            "status": "error",
            "reason": (
                "Image metrics are unavailable. Upload a valid image."
            ),
        }

    width = metrics.get("width")
    height = metrics.get("height")

    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or not isinstance(height, int)
        or isinstance(height, bool)
        or width <= 0
        or height <= 0
    ):
        return {
            "action": "reupload",
            "status": "error",
            "reason": (
                "Valid image dimensions could not be determined."
            ),
        }

    if min(width, height) < 128:
        return {
            "action": "reupload",
            "status": "error",
            "reason": (
                "The image is smaller than 128 pixels in at least "
                "one dimension. Try a higher-resolution image."
            ),
        }

    # 3. Check technical image warnings.
    warnings = metrics.get("warnings", [])

    if warnings:
        return {
            "action": "review_image_quality",
            "status": "warning",
            "reason": (
                "Technical image warnings were detected. Review the "
                "original image before interpreting model outputs."
            ),
            "warnings": warnings,
        }

    # 4. Request model inference when predictions are absent.
    if predictions is None:
        return {
            "action": "run_model",
            "status": "continue",
            "reason": (
                "Image checks passed. Run the X-ray model."
            ),
        }

    if not isinstance(predictions, dict) or not predictions:
        return {
            "action": "stop",
            "status": "error",
            "reason": (
                "The model returned no usable predictions. "
                "Check the model setup."
            ),
        }

    # 5. Request an explanation when one is absent.
    if explanation is None:
        return {
            "action": "generate_explanation",
            "status": "continue",
            "reason": (
                "Model outputs are available. Generate an explanation."
            ),
        }

    if not isinstance(explanation, dict) or not explanation:
        return {
            "action": "stop",
            "status": "error",
            "reason": "The explanation output is invalid.",
        }

    # 6. Request Grad-CAM when no result has been supplied.
    if gradcam is None:
        return {
            "action": "generate_gradcam",
            "status": "continue",
            "reason": (
                "The explanation is ready. Generate a Grad-CAM "
                "visualization."
            ),
        }

    # 7. Continue if Grad-CAM failed or returned no heatmap.
    if (
        not isinstance(gradcam, dict)
        or gradcam.get("heatmap") is None
    ):
        return {
            "action": "continue_without_gradcam",
            "status": "warning",
            "reason": (
                "Grad-CAM is unavailable. Continue presenting the "
                "model output and explanation without the overlay."
            ),
        }

    # 8. All required workflow stages have completed.
    return {
        "action": "present_results",
        "status": "success",
        "reason": (
            "The analysis workflow completed. Model scores are not "
            "a diagnosis and require qualified clinical interpretation."
        ),
    }
