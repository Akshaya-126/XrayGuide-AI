import numpy as np

from src import agent


def _patch_analysis_stages(monkeypatch, gradcam=None):
    monkeypatch.setattr(agent, "enhance_contrast", lambda image: np.zeros((256, 256), dtype=np.uint8))
    monkeypatch.setattr(agent, "predict_findings", lambda image: {"Pneumonia": 0.7})
    monkeypatch.setattr(
        agent,
        "generate_explanation",
        lambda predictions, top_n=1: {"display_name": "Pneumonia"},
    )
    monkeypatch.setattr(
        agent,
        "generate_gradcam",
        lambda image, label: gradcam,
    )


def test_analyze_xray_returns_results_when_all_stages_succeed(monkeypatch):
    heatmap = np.ones((8, 8), dtype=np.float32)
    _patch_analysis_stages(monkeypatch, {"heatmap": heatmap})

    result = agent.analyze_xray(np.zeros((256, 256, 3), dtype=np.uint8))

    assert result["status"] == "success"
    assert result["predictions"] == {"Pneumonia": 0.7}
    assert result["gradcam"]["heatmap"] is heatmap


def test_analyze_xray_keeps_results_when_gradcam_fails(monkeypatch):
    _patch_analysis_stages(monkeypatch)

    def fail_gradcam(image, label):
        raise RuntimeError("Grad-CAM unavailable")

    monkeypatch.setattr(agent, "generate_gradcam", fail_gradcam)
    result = agent.analyze_xray(np.zeros((256, 256, 3), dtype=np.uint8))

    assert result["status"] == "success"
    assert result["predictions"] == {"Pneumonia": 0.7}
    assert result["gradcam"] is None
    assert result["gradcam_error"] == "Grad-CAM unavailable"


def test_analyze_xray_reports_model_failure(monkeypatch):
    monkeypatch.setattr(agent, "enhance_contrast", lambda image: np.zeros((256, 256), dtype=np.uint8))

    def fail_model(image):
        raise RuntimeError("Model unavailable")

    monkeypatch.setattr(agent, "predict_findings", fail_model)
    result = agent.analyze_xray(np.zeros((256, 256, 3), dtype=np.uint8))

    assert result["status"] == "error"
    assert "Model inference failed" in result["message"]
