import numpy as np
import pytest
from src.image_quality import assess_image_quality


def test_metrics_return_expected_keys():
    image = np.full((256, 256, 3), 120, dtype=np.uint8)
    result = assess_image_quality(image)
    assert result["width"] == 256
    assert result["height"] == 256
    assert "brightness" in result
    assert "contrast" in result
    assert "sharpness" in result
    assert "warnings" in result


def test_empty_image_raises():
    with pytest.raises(ValueError):
        assess_image_quality(np.array([]))


def test_grayscale_supported():
    result = assess_image_quality(np.zeros((200, 180), dtype=np.uint8))
    assert result["width"] == 180
    assert result["height"] == 200
