# Evaluation Plan

Do not report performance numbers until tests have actually been run.

## Engineering tests
- Valid RGB image and grayscale image.
- Empty/corrupt upload handling.
- Low-resolution and low-contrast images.
- Model loading and inference on CPU.
- App startup and download report.
- Inference latency and peak memory on the deployment instance.

## Vision evaluation
If claiming finding-detection performance, use a properly licensed, documented held-out dataset and evaluate each supported label with appropriate metrics (e.g., AUROC, sensitivity, specificity, calibration where applicable). Avoid patient-level leakage. Compare predictions to dataset labels with limitations stated. A demo image or a few hand-picked examples is not validation.

## Report failures
Record failed installs, model download problems, corrupt files, unsupported image formats, and model errors. Do not invent results.
