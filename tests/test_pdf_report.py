import numpy as np

from src.pdf_report import build_visual_pdf


def test_build_visual_pdf_returns_pdf_bytes():
    image = np.zeros((64, 64, 3), dtype=np.uint8)

    pdf_bytes = build_visual_pdf(
        original_rgb=image,
        enhanced_rgb=image,
        overlay_rgb=image,
        explanation={"display_name": "Test label"},
    )

    assert pdf_bytes.startswith(b"%PDF")
