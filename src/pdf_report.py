
import io
import os
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image as PILImage

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    PageBreak,
    Table,
    TableStyle,
    KeepTogether,
)


# --------------------------------------------------
# REPORT CONFIGURATION
# --------------------------------------------------

REPORT_TITLE = "XrayGuide AI"
REPORT_SUBTITLE = "Chest X-ray Research Analysis Report"

DISCLAIMER = (
    "For research and educational use only. This report is not a medical "
    "diagnosis and must not be used to rule out a disease, determine "
    "treatment, or decide whether medical care is necessary. AI results "
    "may be inaccurate. A qualified healthcare professional must interpret "
    "the original X-ray alongside symptoms, examination findings, and "
    "medical history."
)

GRADCAM_NOTE = (
    "The Grad-CAM overlay highlights image regions that influenced the "
    "model output. These regions are not confirmed disease locations, "
    "validated lesion boundaries, or proof that a condition is present."
)

IMAGE_NOTE = (
    "Contrast enhancement changes the appearance of the original image. "
    "It does not add medical information or guarantee better model accuracy."
)

EMERGENCY_FALLBACK = (
    "If you experience severe difficulty breathing, severe chest pain, "
    "blue or grey lips, or new confusion, seek emergency medical care "
    "immediately. Do not wait for or rely on this AI result."
)

CLINIC_GUIDANCE_FALLBACK = (
    "Arrange a review with a qualified healthcare professional to interpret "
    "the original X-ray alongside your symptoms and medical history. "
    "This application cannot determine whether a clinic visit is necessary "
    "or how urgently you need care."
)


# --------------------------------------------------
# IMAGE HELPERS
# --------------------------------------------------

def _normalise_image(image):
    """
    Convert an image array into a valid RGB uint8 array.

    Supports grayscale, RGB, and RGBA images.
    """
    if image is None:
        raise ValueError("An image required for the report is missing.")

    array = np.asarray(image)

    if array.size == 0:
        raise ValueError("An empty image cannot be added to the report.")

    if array.ndim == 2:
        array = np.stack([array] * 3, axis=-1)

    elif array.ndim == 3:
        if array.shape[2] == 1:
            array = np.repeat(array, 3, axis=2)

        elif array.shape[2] == 4:
            # Composite transparency onto a white background.
            rgba = PILImage.fromarray(
                np.asarray(array, dtype=np.uint8),
                mode="RGBA",
            )
            background = PILImage.new(
                "RGB",
                rgba.size,
                (255, 255, 255),
            )
            background.paste(rgba, mask=rgba.getchannel("A"))
            array = np.asarray(background)

        elif array.shape[2] != 3:
            raise ValueError(
                "The image must have 1, 3, or 4 channels."
            )

    else:
        raise ValueError(
            "Unsupported image dimensions. Expected a 2D or 3D image."
        )

    if np.issubdtype(array.dtype, np.floating):
        array = np.nan_to_num(
            array,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

        if array.min() >= 0.0 and array.max() <= 1.0:
            array = array * 255.0

    return np.clip(array, 0, 255).astype(np.uint8)


def _image_to_buffer(image):
    """Convert a NumPy image array to an in-memory PNG buffer."""
    rgb_array = _normalise_image(image)

    pil_image = PILImage.fromarray(rgb_array, mode="RGB")

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def _make_report_image(image, max_width, max_height):
    """
    Create a ReportLab image while preserving its aspect ratio.
    """
    buffer = _image_to_buffer(image)

    pil_image = PILImage.open(buffer)
    image_width, image_height = pil_image.size

    scale = min(
        max_width / image_width,
        max_height / image_height,
    )

    report_image = Image(
        buffer,
        width=image_width * scale,
        height=image_height * scale,
    )

    report_image.hAlign = "CENTER"

    return report_image, buffer


# --------------------------------------------------
# TEXT HELPERS
# --------------------------------------------------

def _safe_text(value, fallback="Not provided"):
    """
    Safely convert text into escaped XML for ReportLab Paragraphs.
    """
    if value is None:
        value = fallback

    text = str(value).strip()

    if not text:
        text = fallback

    return escape(text).replace("\n", "<br/>")


def _get_styles():
    """Create the PDF typography styles."""
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#17324D"),
            spaceAfter=8,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ReportSubtitle",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=14,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#526579"),
            spaceAfter=18,
        )
    )

    styles.add(
        ParagraphStyle(
            name="SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#17324D"),
            spaceBefore=10,
            spaceAfter=8,
            keepWithNext=True,
        )
    )

    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=9.5,
            leading=14,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#263445"),
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="ImageCaption",
            parent=styles["Normal"],
            fontName="Helvetica-Oblique",
            fontSize=8,
            leading=11,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#526579"),
            spaceBefore=4,
            spaceAfter=10,
        )
    )

    styles.add(
        ParagraphStyle(
            name="WarningText",
            parent=styles["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=9,
            leading=13,
            textColor=colors.HexColor("#8A2C20"),
        )
    )

    styles.add(
        ParagraphStyle(
            name="FooterText",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=7,
            leading=9,
            textColor=colors.HexColor("#667085"),
        )
    )

    return styles


# --------------------------------------------------
# PAGE HEADER AND FOOTER
# --------------------------------------------------

def _draw_page_elements(canvas, document):
    """Draw a consistent header and footer on every PDF page."""
    canvas.saveState()

    page_width, page_height = A4

    # Header line
    canvas.setStrokeColor(colors.HexColor("#D8E1EA"))
    canvas.setLineWidth(0.6)

    canvas.line(
        document.leftMargin,
        page_height - 15 * mm,
        page_width - document.rightMargin,
        page_height - 15 * mm,
    )

    canvas.setFont("Helvetica-Bold", 8)
    canvas.setFillColor(colors.HexColor("#17324D"))

    canvas.drawString(
        document.leftMargin,
        page_height - 12 * mm,
        REPORT_TITLE,
    )

    # Footer line
    canvas.line(
        document.leftMargin,
        13 * mm,
        page_width - document.rightMargin,
        13 * mm,
    )

    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(colors.HexColor("#667085"))

    canvas.drawString(
        document.leftMargin,
        8.5 * mm,
        "Research and educational use only — not for clinical diagnosis",
    )

    canvas.drawRightString(
        page_width - document.rightMargin,
        8.5 * mm,
        f"Page {document.page}",
    )

    canvas.restoreState()


# --------------------------------------------------
# MAIN PDF GENERATOR
# --------------------------------------------------

def build_visual_pdf(
    original_rgb,
    enhanced_rgb,
    overlay_rgb,
    explanation,
    selected_label=None,
):
    """
    Build an XrayGuide AI PDF report and return its bytes.

    Parameters
    ----------
    original_rgb : numpy.ndarray
        Original chest X-ray in RGB format.

    enhanced_rgb : numpy.ndarray
        Contrast-enhanced X-ray in RGB or grayscale format.

    overlay_rgb : numpy.ndarray
        Grad-CAM overlay blended with the original X-ray.

    explanation : dict
        Output returned by generate_explanation().

    selected_label : str, optional
        Display name of the highest-ranked model finding.

    Returns
    -------
    bytes
        Complete PDF document as bytes, ready for Streamlit download.
    """
    if not isinstance(explanation, dict):
        raise TypeError(
            "explanation must be a dictionary returned by "
            "generate_explanation()."
        )

    # Validate and prepare the image arrays.
    original_rgb = _normalise_image(original_rgb)
    enhanced_rgb = _normalise_image(enhanced_rgb)
    overlay_rgb = _normalise_image(overlay_rgb)

    # Extract explanation fields.
    display_name = (
        selected_label
        or explanation.get("display_name")
        or explanation.get("top_label")
        or "Unavailable"
    )

    summary = explanation.get(
        "summary",
        "No model-generated summary was provided.",
    )

    meaning = explanation.get(
        "meaning",
        "No additional explanation was provided.",
    )

    context = explanation.get(
        "context",
        "The output requires interpretation by a qualified professional.",
    )

    clinic_guidance = explanation.get(
        "clinic_guidance",
        CLINIC_GUIDANCE_FALLBACK,
    )

    emergency_warning = explanation.get(
        "emergency_warning",
        EMERGENCY_FALLBACK,
    )

    disclaimer = explanation.get(
        "disclaimer",
        DISCLAIMER,
    )

    # Prepare an in-memory PDF.
    pdf_buffer = io.BytesIO()

    document = SimpleDocTemplate(
        pdf_buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=21 * mm,
        bottomMargin=19 * mm,
        title=REPORT_TITLE,
        author="XrayGuide AI",
        subject="Research chest X-ray image analysis",
    )

    styles = _get_styles()
    story = []

    # --------------------------------------------------
    # TITLE
    # --------------------------------------------------

    story.append(Spacer(1, 5 * mm))

    story.append(
        Paragraph(
            _safe_text(REPORT_TITLE),
            styles["ReportTitle"],
        )
    )

    story.append(
        Paragraph(
            _safe_text(REPORT_SUBTITLE),
            styles["ReportSubtitle"],
        )
    )

    story.append(
        Paragraph(
            "Important: this report contains experimental AI output. "
            "It is not a confirmed medical diagnosis.",
            styles["WarningText"],
        )
    )

    story.append(Spacer(1, 6 * mm))

    # --------------------------------------------------
    # ORIGINAL X-RAY
    # --------------------------------------------------

    story.append(
        Paragraph(
            "1. Original chest X-ray",
            styles["SectionHeading"],
        )
    )

    original_image, original_buffer = _make_report_image(
        original_rgb,
        max_width=165 * mm,
        max_height=105 * mm,
    )

    story.append(original_image)

    story.append(
        Paragraph(
            "Original image supplied for this analysis.",
            styles["ImageCaption"],
        )
    )

    story.append(
        Paragraph(
            f"Image dimensions: {original_rgb.shape[1]} × "
            f"{original_rgb.shape[0]} pixels.",
            styles["BodyTextCustom"],
        )
    )

    # --------------------------------------------------
    # MODEL FINDING
    # --------------------------------------------------

    story.append(
        Paragraph(
            "2. Top-ranked model finding",
            styles["SectionHeading"],
        )
    )

    finding_data = [
        [
            Paragraph(
                "<b>Model output</b>",
                styles["BodyTextCustom"],
            ),
            Paragraph(
                _safe_text(display_name),
                styles["BodyTextCustom"],
            ),
        ],
        [
            Paragraph(
                "<b>Summary</b>",
                styles["BodyTextCustom"],
            ),
            Paragraph(
                _safe_text(summary),
                styles["BodyTextCustom"],
            ),
        ],
    ]

    finding_table = Table(
        finding_data,
        colWidths=[37 * mm, 128 * mm],
        hAlign="LEFT",
    )

    finding_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (0, -1),
                    colors.HexColor("#EEF3F8"),
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#263445"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.HexColor("#D8E1EA"),
                ),
                (
                    "INNERGRID",
                    (0, 0),
                    (-1, -1),
                    0.4,
                    colors.HexColor("#D8E1EA"),
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP",
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    7,
                ),
            ]
        )
    )

    story.append(finding_table)
    story.append(Spacer(1, 4 * mm))

    story.append(
        Paragraph(
            "<b>What does this mean?</b>",
            styles["BodyTextCustom"],
        )
    )

    story.append(
        Paragraph(
            _safe_text(meaning),
            styles["BodyTextCustom"],
        )
    )

    story.append(
        Paragraph(
            "<b>Important context</b>",
            styles["BodyTextCustom"],
        )
    )

    story.append(
        Paragraph(
            _safe_text(context),
            styles["BodyTextCustom"],
        )
    )

    story.append(
        Paragraph(
            "The highest-ranked output is not a calibrated probability "
            "of disease. It does not establish or exclude a diagnosis.",
            styles["BodyTextCustom"],
        )
    )

    # --------------------------------------------------
    # CONTRAST-ENHANCED IMAGE
    # --------------------------------------------------

    story.append(PageBreak())

    story.append(
        Paragraph(
            "3. Contrast-enhanced image",
            styles["SectionHeading"],
        )
    )

    enhanced_image, enhanced_buffer = _make_report_image(
        enhanced_rgb,
        max_width=165 * mm,
        max_height=115 * mm,
    )

    story.append(enhanced_image)

    story.append(
        Paragraph(
            IMAGE_NOTE,
            styles["ImageCaption"],
        )
    )

    # --------------------------------------------------
    # GRAD-CAM OVERLAY
    # --------------------------------------------------

    story.append(
        Paragraph(
            "4. Grad-CAM visual explanation",
            styles["SectionHeading"],
        )
    )

    overlay_image, overlay_buffer = _make_report_image(
        overlay_rgb,
        max_width=165 * mm,
        max_height=115 * mm,
    )

    story.append(overlay_image)

    story.append(
        Paragraph(
            GRADCAM_NOTE,
            styles["ImageCaption"],
        )
    )

    # --------------------------------------------------
    # CLINIC GUIDANCE
    # --------------------------------------------------

    story.append(
        Paragraph(
            "5. Suggested next step",
            styles["SectionHeading"],
        )
    )

    guidance_table = Table(
        [
            [
                Paragraph(
                    _safe_text(clinic_guidance),
                    styles["BodyTextCustom"],
                )
            ]
        ],
        colWidths=[165 * mm],
    )

    guidance_table.setStyle(
        TableStyle(
            [
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, -1),
                    colors.HexColor("#EEF5FB"),
                ),
                (
                    "BOX",
                    (0, 0),
                    (-1, -1),
                    0.7,
                    colors.HexColor("#A8C6DF"),
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    9,
                ),
            ]
        )
    )

    story.append(guidance_table)
    story.append(Spacer(1, 4 * mm))

    story.append(
        Paragraph(
            "<b>Emergency warning</b>",
            styles["WarningText"],
        )
    )

    story.append(
        Paragraph(
            _safe_text(emergency_warning),
            styles["BodyTextCustom"],
        )
    )

    # --------------------------------------------------
    # LIMITATIONS AND DISCLAIMER
    # --------------------------------------------------

    story.append(
        Paragraph(
            "6. Limitations and disclaimer",
            styles["SectionHeading"],
        )
    )

    story.append(
        Paragraph(
            _safe_text(disclaimer),
            styles["BodyTextCustom"],
        )
    )

    story.append(
        Paragraph(
            _safe_text(DISCLAIMER),
            styles["BodyTextCustom"],
        )
    )

    story.append(
        Spacer(1, 3 * mm)
    )

    story.append(
        Paragraph(
            "This report is automatically generated by XrayGuide AI. "
            "It should not replace a radiologist's report or a clinician's "
            "assessment.",
            styles["FooterText"],
        )
    )

    # --------------------------------------------------
    # BUILD PDF
    # --------------------------------------------------

    document.build(
        story,
        onFirstPage=_draw_page_elements,
        onLaterPages=_draw_page_elements,
    )

    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.getvalue()

    if not pdf_bytes.startswith(b"%PDF"):
        raise RuntimeError("PDF generation failed.")

    return pdf_bytes
