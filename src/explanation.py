
import math


# --------------------------------------------------
# Plain-language explanations for supported labels
# --------------------------------------------------
LABEL_INFO = {
    "Atelectasis": {
        "display_name": "Atelectasis",
        "meaning": (
            "A part of the lung may be partially collapsed "
            "or may not be fully expanded."
        ),
        "context": (
            "This model label may be associated with different "
            "imaging patterns. It does not confirm atelectasis."
        ),
    },
    "Consolidation": {
        "display_name": "Consolidation",
        "meaning": (
            "An area of the lung may appear denser than usual, "
            "sometimes because air spaces contain fluid or other material."
        ),
        "context": (
            "There can be several causes of this appearance. "
            "The model output alone cannot identify the cause "
            "or confirm a finding."
        ),
    },
    "Infiltration": {
        "display_name": "Infiltration",
        "meaning": (
            "A broad, nonspecific term historically used to describe "
            "an area of increased opacity in the lung."
        ),
        "context": (
            "This term is not a diagnosis and may be used differently "
            "across imaging datasets."
        ),
    },
    "Pneumothorax": {
        "display_name": "Pneumothorax",
        "meaning": (
            "A pneumothorax occurs when air collects in the space "
            "between the lung and chest wall, potentially causing "
            "the lung to collapse partly or completely."
        ),
        "context": (
            "This can be a serious condition, but the AI output does "
            "not establish that it is present. Severe breathing "
            "difficulty or severe chest pain requires emergency care."
        ),
    },
    "Edema": {
        "display_name": "Edema",
        "meaning": (
            "Edema means swelling caused by fluid. In a chest X-ray "
            "context, this label may relate to fluid in lung tissues."
        ),
        "context": (
            "The model cannot confirm edema or establish its cause. "
            "Professional interpretation is necessary."
        ),
    },
    "Emphysema": {
        "display_name": "Emphysema",
        "meaning": (
            "Emphysema is a chronic lung condition involving damage "
            "to the air sacs and reduced elasticity of the lungs."
        ),
        "context": (
            "This model output cannot diagnose emphysema. "
            "Clinical history and professional interpretation are needed."
        ),
    },
    "Fibrosis": {
        "display_name": "Fibrosis",
        "meaning": (
            "Fibrosis refers to scar-like tissue changes. "
            "In the lungs, these changes can affect normal tissue."
        ),
        "context": (
            "The model output does not confirm lung fibrosis "
            "or establish its cause."
        ),
    },
    "Effusion": {
        "display_name": "Pleural effusion",
        "meaning": (
            "A pleural effusion is fluid collecting in the space "
            "between the lung and the chest wall."
        ),
        "context": (
            "The model output does not confirm fluid or identify "
            "its cause. A healthcare professional must interpret the X-ray."
        ),
    },
    "Pneumonia": {
        "display_name": "Pneumonia",
        "meaning": (
            "Pneumonia is an infection that inflames air sacs in "
            "one or both lungs."
        ),
        "context": (
            "Symptoms, medical history, examination, and professional "
            "interpretation are needed. This AI output cannot confirm "
            "or exclude pneumonia."
        ),
    },
    "Pleural_Thickening": {
        "display_name": "Pleural thickening",
        "meaning": (
            "The pleura, the thin lining around the lungs, "
            "may appear thicker than usual."
        ),
        "context": (
            "There can be multiple explanations for this appearance. "
            "The model output is not a confirmed finding."
        ),
    },
    "Cardiomegaly": {
        "display_name": "Cardiomegaly",
        "meaning": (
            "Cardiomegaly means an enlarged heart or an enlarged-looking "
            "heart silhouette on imaging."
        ),
        "context": (
            "Image positioning and technique can affect the heart's "
            "appearance. This output cannot confirm heart enlargement."
        ),
    },
    "Nodule": {
        "display_name": "Nodule",
        "meaning": (
            "A nodule is a small, rounded area that may appear "
            "on medical imaging."
        ),
        "context": (
            "This output does not confirm a nodule, its location, "
            "or its nature. Professional review is required."
        ),
    },
    "Mass": {
        "display_name": "Mass",
        "meaning": (
            "A mass is a larger area or growth described as "
            "an abnormal-looking structure on imaging."
        ),
        "context": (
            "This output does not confirm that a mass is present "
            "or indicate whether a finding is benign or malignant."
        ),
    },
    "Hernia": {
        "display_name": "Hernia",
        "meaning": (
            "A hernia occurs when tissue or an organ pushes through "
            "an opening or weakened area. Some hernias can affect "
            "the chest region."
        ),
        "context": (
            "This model label is not confirmation of a hernia. "
            "Professional image interpretation is necessary."
        ),
    },
    "Lung Lesion": {
        "display_name": "Lung lesion",
        "meaning": (
            "A lesion is a general term for an area of tissue "
            "that looks different from surrounding tissue."
        ),
        "context": (
            "The term does not specify a cause and does not by itself "
            "mean cancer. This model output is not a confirmed finding."
        ),
    },
    "Fracture": {
        "display_name": "Fracture",
        "meaning": (
            "A fracture is a break or crack in a bone, "
            "which may include a rib."
        ),
        "context": (
            "This output cannot confirm a fracture. A healthcare "
            "professional should interpret the image alongside "
            "symptoms and injury history."
        ),
    },
    "Lung Opacity": {
        "display_name": "Lung opacity",
        "meaning": (
            "An opacity is an area that appears whiter or denser "
            "than expected on an X-ray."
        ),
        "context": (
            "Lung opacity is a nonspecific imaging description with "
            "many possible causes. This output does not identify "
            "a cause or confirm a finding."
        ),
    },
    "Enlarged Cardiomediastinum": {
        "display_name": "Enlarged cardiomediastinum",
        "meaning": (
            "The cardiomediastinum is the central chest area containing "
            "the heart and nearby structures. It may appear enlarged "
            "on an X-ray."
        ),
        "context": (
            "Positioning and image technique can affect this appearance. "
            "The model output cannot confirm enlargement or explain its cause."
        ),
    },
}


# --------------------------------------------------
# General safety guidance
# --------------------------------------------------
CLINIC_GUIDANCE = (
    "Arrange a review with a qualified healthcare professional to "
    "interpret the original X-ray alongside your symptoms and medical "
    "history. This app cannot determine whether a clinic visit is "
    "necessary or how urgently you need care."
)

EMERGENCY_WARNING = (
    "If you have severe difficulty breathing, severe chest pain, "
    "blue or grey lips, or new confusion, seek emergency medical care "
    "immediately. Do not wait for or rely on this AI result."
)

DISCLAIMER = (
    "For research and educational use only. This tool is not a diagnosis, "
    "does not rule out disease, and is not validated for clinical triage. "
    "Grad-CAM highlights image regions that influence the model output; "
    "it does not confirm the location of a disease."
)


# --------------------------------------------------
# Generate the highest-ranked label explanation
# --------------------------------------------------
def generate_explanation(predictions, top_n=1):
    """
    Generate a plain-language explanation for the highest-ranked
    model output.

    Parameters
    ----------
    predictions : dict
        Dictionary mapping model labels to numeric output scores.

    top_n : int
        Retained for compatibility with earlier code. This function
        intentionally returns only the highest-ranked label.

    Returns
    -------
    dict
        Label, explanation, clinic guidance, emergency warning,
        and disclaimer.

    Important
    ---------
    Model scores are not calibrated patient-specific probabilities,
    confirmed findings, or a validated basis for clinical triage.
    """

    # Handle a missing or empty prediction dictionary.
    if not isinstance(predictions, dict) or not predictions:
        return {
            "status": "unavailable",
            "top_label": None,
            "display_name": "No result available",
            "model_score": None,
            "summary": (
                "The model did not return a usable result."
            ),
            "meaning": (
                "No explanation is available because no valid "
                "model output was received."
            ),
            "context": (
                "Do not use this result to make healthcare decisions."
            ),
            "clinic_guidance": CLINIC_GUIDANCE,
            "emergency_warning": EMERGENCY_WARNING,
            "disclaimer": DISCLAIMER,
        }

    # Keep only numeric, finite scores.
    valid_predictions = []

    for label, score in predictions.items():
        try:
            numeric_score = float(score)

            if math.isfinite(numeric_score):
                valid_predictions.append(
                    (str(label), numeric_score)
                )

        except (TypeError, ValueError):
            continue

    # Handle invalid scores.
    if not valid_predictions:
        return {
            "status": "unavailable",
            "top_label": None,
            "display_name": "No usable model output",
            "model_score": None,
            "summary": (
                "The model did not return a usable numeric score."
            ),
            "meaning": (
                "An explanation cannot be generated from the "
                "available output."
            ),
            "context": (
                "Do not interpret missing or invalid scores as "
                "evidence that the X-ray is normal."
            ),
            "clinic_guidance": CLINIC_GUIDANCE,
            "emergency_warning": EMERGENCY_WARNING,
            "disclaimer": DISCLAIMER,
        }

    # Find the highest-ranked output.
    # This selects a model label, not a confirmed medical finding.
    top_label, top_score = max(
        valid_predictions,
        key=lambda item: item[1],
    )

    # Look up the label explanation.
    label_info = LABEL_INFO.get(top_label, {})

    display_name = label_info.get(
        "display_name",
        top_label.replace("_", " "),
    )

    meaning = label_info.get(
        "meaning",
        (
            "This is a label used by the AI model to describe "
            "a possible imaging pattern."
        ),
    )

    context = label_info.get(
        "context",
        (
            "This output is not a confirmed finding. A qualified "
            "healthcare professional must interpret the original "
            "X-ray alongside symptoms and medical history."
        ),
    )

    # Return only one label and its explanation.
    return {
        "status": "Model output available",
        "top_label": top_label,
        "display_name": display_name,
        "model_score": top_score,
        "summary": (
            f"The highest-ranked model output is {display_name}. "
            "This is an AI-generated result, not a confirmed finding."
        ),
        "meaning": meaning,
        "context": context,
        "clinic_guidance": CLINIC_GUIDANCE,
        "emergency_warning": EMERGENCY_WARNING,
        "disclaimer": DISCLAIMER,
    }
