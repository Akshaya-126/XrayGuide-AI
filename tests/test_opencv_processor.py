import numpy as np
import pytest
from src.opencv_processor import enhance_contrast


def test_clahe_returns_grayscale_image():
    image = np.random.default_rng(7).integers(0, 256, (256, 256, 3), dtype=np.uint8)
    result = enhance_contrast(image)
    assert result.shape == (256, 256)
    assert result.dtype == np.uint8


def test_empty_image_raises():
    with pytest.raises(ValueError):
        enhance_contrast(np.array([]))
