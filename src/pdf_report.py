
import io
from xml.sax.saxutils import escape

import cv2
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
    """Convert grayscale, RGB or RGBA image arrays to RGB uint8."""

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
            rgba = PILImage.fromarray(
                np.asarray(array, dtype=np.uint8)
            )

            rgba = rgba.convert("RGBA")

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
    """Convert a NumPy image to an in-memory PNG buffer."""

    rgb_array = _normalise_image(image)
    pil_image = PILImage.fromarray(rgb_array)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    buffer.seek(0)

    return buffer


def _make_report_image(image, max_width, max_height):
    """Create a ReportLab image while preserving aspect ratio."""

    buffer = _image_to_buffer(image)

    with PILImage.open(buffer) as pil_image:
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


def _calculate_measurements(image, file_size_bytes=None):
    """
    Calculate image dimensions and grayscale pixel statistics.

    These measurements describe the image, not anatomy or disease.
    """

    array = _normalise_image(image)

    grayscale = cv2.cvtColor(
        array,
        cv2.COLOR_RGB2GRAY,
    )

    height, width = grayscale.shape

    mean_intensity = float(np.mean(grayscale))
    std_intensity = float(np.std(grayscale))

    percentile_5, percentile_95 = np.percentile(
        grayscale,
        [5, 95],
    )

    contrast_range = float(percentile_95 - percentile_5)

    measurements = [
        ("Image width", f"{width:,} px"),
        ("Image height", f"{height:,} px"),
    ]

    if file_size_bytes is not None:
        if file_size_bytes >= 1024 * 1024:
            size_text = (
                f"{file_size_bytes / (1024 * 1024):.2f} MB"
            )
        else:
            size_text = f"{file_size_bytes / 1024:.2f} KB"

        measurements.append(("Original file size", size_text))

    measurements.extend([
        (
            "Mean grayscale intensity",
            f"{mean_intensity:.2f} / 255",
        ),
        (
            "Grayscale standard deviation",
            f"{std_intensity:.2f}",
        ),
        (
            "Contrast estimate (P95 - P5)",
            f"{contrast_range:.2f} / 255",
        ),
    ])

    return measurements


# --------------------------------------------------
# TEXT AND TABLE HELPERS
# --------------------------------------------------

def _safe_text(value, fallback="Not provided"):
    """Escape text for ReportLab Paragraphs."""

    if value is None:
        value = fallback

    text = str(value).strip()

    if not text:
        text = fallback

    return escape(text).replace("\n", "<br/>")


def _measurement_table(measurements, styles):
    """Build a formatted table containing image measurements."""

    rows = [
        [
            Paragraph("<b>Measurement</b>", styles["TableText"]),
            Paragraph("<b>Value</b>", styles["TableText"]),
        ]
    ]

    for label, value in measurements:
        rows.append([
            Paragraph(_safe_text(label), styles["TableText"]),
            Paragraph(_safe_text(value), styles["TableText"]),
        ])

    table = Table(
        rows,
        colWidths=[82 * mm, 83 * mm],
        repeatRows=1,
        hAlign="LEFT",
    )

    table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.HexColor("#E7EEF5"),
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, -1),
                colors.HexColor("#263445"),
            ),
            (
                "GRID",
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
                7,
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                7,
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5,
            ),
        ])
    )

    return table


def _get_styles():
    """Create PDF typography styles."""

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
            fontSize=9,
            leading=13,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#263445"),
            spaceAfter=7,
        )
    )

    styles.add(
        ParagraphStyle(
            name="TableText",
            parent=styles["BodyText"],
            fontName="Helvetica",
            fontSize=8.5,
            leading=11,
            textColor=colors.HexColor("#263445"),
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
            spaceAfter=8,
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
            spaceAfter=7,
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
    Build a PDF report and return its bytes.

    The explanation context can include workflow and Grad-CAM details.
    The image measurements are calculated directly from the image arrays.
    """

    if not isinstance(explanation, dict):
        raise TypeError("explanation must be a dictionary.")

    original_rgb = _normalise_image(original_rgb)
    enhanced_rgb = _normalise_image(enhanced_rgb)
    overlay_rgb = _normalise_image(overlay_rgb)

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

    # Calculate measurements independently for each image.
    # The original uploaded file size is supplied through explanation
    # context by the app, when available.
    original_file_size = explanation.get("original_file_size_bytes")

    original_measurements = _calculate_measurements(
        original_rgb,
        file_size_bytes=original_file_size,
    )

    enhanced_measurements = _calculate_measurements(
        enhanced_rgb,
    )

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

    story.append(Spacer(1, 4 * mm))

    # --------------------------------------------------
    # ORIGINAL X-RAY AND MEASUREMENTS
    # --------------------------------------------------

    story.append(
        Paragraph(
            "1. Original chest X-ray",
            styles["SectionHeading"],
        )
    )

    original_image, original_buffer = _make_report_image(
        original_rgb,
        max_width=145 * mm,
        max_height=85 * mm,
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
            "Original X-ray measurements",
            styles["SectionHeading"],
        )
    )

    story.append(
        _measurement_table(original_measurements, styles)
    )

    story.append(Spacer(1, 5 * mm))

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
            Paragraph("<b>Model output</b>", styles["TableText"]),
            Paragraph(_safe_text(display_name), styles["TableText"]),
        ],
        [
            Paragraph("<b>Summary</b>", styles["TableText"]),
            Paragraph(_safe_text(summary), styles["TableText"]),
        ],
    ]

    finding_table = Table(
        finding_data,
        colWidths=[37 * mm, 128 * mm],
        hAlign="LEFT",
    )

    finding_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, -1),
                colors.HexColor("#EEF3F8"),
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
        ])
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
    # ENHANCED X-RAY AND MEASUREMENTS
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
        max_width=145 * mm,
        max_height=80 * mm,
    )

    story.append(enhanced_image)

    story.append(
        Paragraph(
            IMAGE_NOTE,
            styles["ImageCaption"],
        )
    )

    story.append(
        Paragraph(
            "Enhanced X-ray measurements",
            styles["SectionHeading"],
        )
    )

    story.append(
        _measurement_table(enhanced_measurements, styles)
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
        max_width=145 * mm,
        max_height=80 * mm,
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
        [[
            Paragraph(
                _safe_text(clinic_guidance),
                styles["BodyTextCustom"],
            )
        ]],
        colWidths=[165 * mm],
    )

    guidance_table.setStyle(
        TableStyle([
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
        ])
    )

    story.append(guidance_table)
    story.append(Spacer(1, 4 * mm))

    story.append(
        Paragraph(
            "Emergency warning",
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

    story.append(Spacer(1, 3 * mm))

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
