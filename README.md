# XrayGuide AI

A research prototype that combines OpenCV 5 technical image analysis with a pretrained chest X-ray model and a transparent workflow controller.

## Features
- Technical image indicators: dimensions, brightness, contrast and Laplacian sharpness indicator.
- Optional CLAHE image enhancement for visual comparison.
- TorchXRayVision DenseNet-121 pretrained model outputs.
- Plot of raw model scores and JSON technical summary.
- Rule-based workflow controller and basic automated tests.
- Docker configuration intended for AWS EC2 deployment.

## Safety and intended use
This project is for education and software prototyping only. It is not a medical device and must not be used to diagnose or rule out disease, determine treatment, or replace a qualified healthcare professional. Model outputs may be biased, uncalibrated, and unreliable for images unlike its training data. No clinical validation is provided by this project.

## Requirements
- Python 3.11 recommended.
- Internet access for initial dependency installation and pretrained model weights download.
- CPU can run inference, but performance depends on the machine. A dedicated GPU is not required for initial testing.

## Local setup (Windows CMD)
```bat
python -m venv venv
venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local Streamlit URL printed in the terminal. First model inference may take longer because the pretrained weights may need to download.

## Run tests
```bat
python -m pytest
```

## Model details
The app uses TorchXRayVision `DenseNet(weights="densenet121-res224-all")`. Inspect the package/model documentation and license before distribution. The labels come from the loaded model at runtime. The scores are raw model outputs and must not be treated as calibrated individual disease probabilities.

## OpenCV version note
The requirements file pins `opencv-python==5.0.0.93`, as intended for this project. If that version is unavailable for your Python/platform or conflicts with another dependency, verify the published package compatibility before substituting a version; do not silently claim OpenCV 5 if another major version is installed.

## AWS EC2 deployment
1. Create a suitable Linux EC2 instance and configure billing alerts.
2. Restrict the security group to the required source IPs; avoid opening Streamlit to the whole internet unless you have a secure access plan.
3. Install Docker, clone this repository and run `bash deployment/deploy_ec2.sh`.
4. Confirm the app works from the intended browser.
5. Stop or terminate the instance when it is not needed. Verify current AWS pricing before launch.

## Privacy
The app processes uploads in memory and does not intentionally save them. Deployment logs, platform settings, and user actions can still affect privacy. Do not upload identifiable patient images. Do not enable persistent image storage without an explicit privacy and security design.

## Project structure
See the repository folders `src/`, `config/`, `tests/`, `docs/`, `deployment/`, and `assets/`.
