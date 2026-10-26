from datetime import datetime, timezone


def build_report(metrics: dict, predictions: dict, decision: dict) -> dict:
    """Create a JSON-serializable technical summary."""
    ranked = sorted(
        [{"label": str(label), "model_score": float(score)}
         for label, score in predictions.items()],
        key=lambda row: row["model_score"],
        reverse=True,
    )
    return {
        "application": "XrayGuide AI",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "notice": (
            "Research prototype only. Model outputs are not diagnoses or calibrated "
            "patient-specific disease probabilities. Seek professional interpretation."
        ),
        "technical_image_metrics": metrics,
        "model_outputs": ranked,
        "agent_review": decision,
    }
