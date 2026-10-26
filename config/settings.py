"""Central defaults. Thresholds are engineering heuristics, not clinical standards."""

APP_NAME = "XrayGuide AI"
MIN_IMAGE_SIDE_PIXELS = 128
BRIGHTNESS_LOW_WARNING = 25
BRIGHTNESS_HIGH_WARNING = 230
LOW_CONTRAST_WARNING = 15
LOW_SHARPNESS_WARNING = 20

MODEL_NAME = "TorchXRayVision DenseNet-121"
MODEL_WEIGHTS = "densenet121-res224-all"
MODEL_CACHE_SIZE = 1
