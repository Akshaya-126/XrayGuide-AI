from src.agent_controller import decide_next_action


def test_missing_metrics_requests_reupload():
    result = decide_next_action({})
    assert result["action"] == "reupload"


def test_quality_warning_requests_review():
    metrics = {"width": 512, "height": 512, "warnings": ["Image contrast is low."]}
    result = decide_next_action(metrics)
    assert result["action"] == "review_image_quality"


def test_clean_image_with_predictions_requests_explanation():
    metrics = {"width": 512, "height": 512, "warnings": []}
    result = decide_next_action(metrics, {"Finding": 0.2})
    assert result["action"] == "generate_explanation"
