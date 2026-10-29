# XrayGuide AI Architecture

## Flow
1. Streamlit accepts a chest X-ray image.
2. OpenCV computes basic technical image indicators.
3. TorchXRayVision DenseNet-121 produces outputs for the labels supported by its pretrained weights.
4. A transparent rule-based controller selects a workflow action based on technical metrics and whether model outputs exist.
5. The dashboard visualizes raw model outputs and offers a downloadable PDF report.
6. If enabled and configured, the original image is uploaded to private S3 storage. Generated PDFs are not uploaded.
7. AWS EC2 is the planned hosting target.

## Components
- `src/image_quality.py`: technical metrics; thresholds are heuristics.
- `src/opencv_processor.py`: optional CLAHE visualization.
- `src/xray_model.py`: model loading and inference.
- `src/agent_controller.py`: bounded, explainable rule-based control.
- `src/visualization.py`: chart of model outputs.
- `src/report_generator.py`: JSON summary.

## Limitations
The model is a research model, may be biased by training data and acquisition differences, and is not clinically validated by this project. Scores are not calibrated patient-specific probabilities. CLAHE is optional and does not establish clinical improvement. The system must not be used to diagnose, rule out disease, or decide treatment.
