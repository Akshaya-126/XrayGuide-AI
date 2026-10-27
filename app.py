
import hashlib
import io
import traceback

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from src.agent import analyze_xray
from src.pdf_report import build_visual_pdf
from s3_storage import upload_bytes


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="XrayGuide AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1250px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

    .main-title {
        font-size: 2.15rem;
        font-weight: 700;
        margin-bottom: 0.15rem;
    }

    .subtitle {
        color: #64748b;
        margin-bottom: 1.2rem;
    }

    .section-heading {
        font-size: 1.35rem;
        font-weight: 650;
        margin-top: 1.2rem;
        margin-bottom: 0.6rem;
    }

    .measurement-heading {
        font-size: 1.05rem;
        font-weight: 650;
        margin-top: 0.9rem;
        margin-bottom: 0.5rem;
    }

    div[data-testid="stImage"] img {
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="main-title">🩻 XrayGuide AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    'Chest X-ray analysis and visual model explanation'
    '</div>',
    unsafe_allow_html=True,
)

st.warning(
    "Research and educational use only. This prototype is not a "
    "diagnostic or medical triage tool. It cannot determine whether "
    "or how urgently you need medical care."
)


# ============================================================
# IMAGE HELPERS
# ============================================================

def calculate_image_measurements(image_rgb, file_size_bytes=None):
    """
    Calculate basic image dimensions and pixel-intensity measurements.

    These are image-processing measurements, not clinical measurements.
    """

    image_rgb = np.asarray(image_rgb)

    if image_rgb.ndim == 2:
        grayscale = image_rgb
    elif image_rgb.ndim == 3 and image_rgb.shape[2] == 3:
        grayscale = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError(
            "Unsupported image format for measurement calculation."
        )

    height, width = grayscale.shape[:2]

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
    ]

    if file_size_bytes is not None:
        if file_size_bytes >= 1024 * 1024:
            file_size = (
                f"{file_size_bytes / (1024 * 1024):.2f} MB"
            )
        else:
            file_size = f"{file_size_bytes / 1024:.2f} KB"

        measurements.insert(2, ("Original file size", file_size))

    return measurements


def display_image_measurements(image_rgb, file_size_bytes=None):
    """Display image measurements in a compact Streamlit layout."""

    measurements = calculate_image_measurements(
        image_rgb,
        file_size_bytes=file_size_bytes,
    )

    for start in range(0, len(measurements), 2):
        measurement_columns = st.columns(2)

        for column, item in zip(
            measurement_columns,
            measurements[start:start + 2],
        ):
            label, value = item

            with column:
                st.markdown(f"**{label}**")
                st.write(value)


def create_gradcam_images(original_rgb, heatmap):
    """Create a Grad-CAM heatmap overlay and relative activation map."""

    activation_map = np.asarray(heatmap, dtype=np.float32)

    if activation_map.ndim == 3:
        if activation_map.shape[2] == 1:
            activation_map = activation_map[:, :, 0]
        else:
            activation_map = cv2.cvtColor(
                activation_map,
                cv2.COLOR_RGB2GRAY,
            )

    if activation_map.ndim != 2:
        raise ValueError("Grad-CAM must return a 2D activation map.")

    activation_map = np.nan_to_num(
        activation_map,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    activation_map = cv2.normalize(
        activation_map,
        None,
        alpha=0,
        beta=1,
        norm_type=cv2.NORM_MINMAX,
    )

    activation_map = cv2.resize(
        activation_map,
        (original_rgb.shape[1], original_rgb.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )

    heatmap_uint8 = np.clip(
        activation_map * 255,
        0,
        255,
    ).astype(np.uint8)

    heatmap_bgr = cv2.applyColorMap(
        heatmap_uint8,
        cv2.COLORMAP_JET,
    )

    heatmap_rgb = cv2.cvtColor(
        heatmap_bgr,
        cv2.COLOR_BGR2RGB,
    )

    overlay_rgb = cv2.addWeighted(
        original_rgb,
        0.60,
        heatmap_rgb,
        0.40,
        0,
    )

    return overlay_rgb, activation_map


def describe_influential_region(activation_map):
    """Describe the grid area with the highest mean relative activation."""

    height, width = activation_map.shape

    row_names = ["upper", "middle", "lower"]
    column_names = ["left", "centre", "right"]

    row_edges = np.linspace(0, height, 4, dtype=int)
    column_edges = np.linspace(0, width, 4, dtype=int)

    region_means = []

    for row in range(3):
        for column in range(3):
            region = activation_map[
                row_edges[row]:row_edges[row + 1],
                column_edges[column]:column_edges[column + 1],
            ]

            if region.size:
                region_means.append(
                    (
                        float(np.mean(region)),
                        row_names[row],
                        column_names[column],
                    )
                )

    if not region_means:
        return "The influential image region could not be estimated."

    _, vertical, horizontal = max(
        region_means,
        key=lambda item: item[0],
    )

    return (
        f"The highest mean relative activation in this coarse grid "
        f"appears in the {vertical}, {horizontal} part of the image "
        f"(viewer's perspective). This describes model activation, "
        f"not a confirmed location of disease."
    )


# ============================================================
# AGENT WORKFLOW
# ============================================================

def analyse_uploaded_xray(original_rgb):
    """Run the agent and prepare the UI and PDF report data."""

    agent_result = analyze_xray(original_rgb)

    if not isinstance(agent_result, dict):
        raise ValueError("The agent returned an unexpected result.")

    if agent_result.get("status") != "success":
        raise ValueError(
            agent_result.get(
                "message",
                "The agent could not complete the analysis.",
            )
        )

    explanation = agent_result.get("explanation")
    predictions = agent_result.get("predictions")
    gradcam_result = agent_result.get("gradcam")

    if not isinstance(explanation, dict):
        raise ValueError("The agent returned an invalid explanation.")

    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("The agent returned no model predictions.")

    enhanced_image = agent_result.get("enhanced_image")

    if enhanced_image is None:
        raise ValueError("The agent did not return an enhanced image.")

    enhanced_image = np.asarray(enhanced_image)

    if enhanced_image.ndim == 2:
        enhanced_rgb = cv2.cvtColor(
            enhanced_image,
            cv2.COLOR_GRAY2RGB,
        )
    elif enhanced_image.ndim == 3 and enhanced_image.shape[2] == 3:
        enhanced_rgb = enhanced_image
    else:
        raise ValueError(
            "The agent returned an unsupported enhanced image format."
        )

    overlay_rgb = None
    activation_map = None
    gradcam_error = agent_result.get("gradcam_error")

    influential_region = (
        "The influential region could not be estimated because "
        "the Grad-CAM visualization is unavailable."
    )

    if isinstance(gradcam_result, dict):
        heatmap = gradcam_result.get("heatmap")

        if heatmap is not None:
            try:
                overlay_rgb, activation_map = create_gradcam_images(
                    original_rgb,
                    heatmap,
                )

                influential_region = describe_influential_region(
                    activation_map
                )

            except Exception as exc:
                gradcam_error = str(exc)
        else:
            gradcam_error = (
                gradcam_error or "The Grad-CAM heatmap is missing."
            )
    else:
        gradcam_error = (
            gradcam_error or "The Grad-CAM result is unavailable."
        )

    decision = agent_result.get("decision", {})

    if not isinstance(decision, dict):
        decision = {}

    return {
        "original_rgb": original_rgb,
        "enhanced_rgb": enhanced_rgb,
        "predictions": predictions,
        "explanation": explanation,
        "gradcam": gradcam_result,
        "overlay_rgb": overlay_rgb,
        "activation_map": activation_map,
        "influential_region": influential_region,
        "gradcam_error": gradcam_error,
        "decision": decision,
    }


# ============================================================
# IMAGE UPLOAD
# ============================================================

st.markdown(
    '<div class="section-heading">Upload your chest X-ray</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Choose a PNG or JPEG image",
    type=["png", "jpg", "jpeg"],
    help="Analysis starts automatically after you upload an image.",
)

if uploaded_file is None:
    for key in (
        "xray_result",
        "xray_image_key",
        "xray_error",
        "xray_traceback",
        "xray_s3_key",
        "xray_s3_error",
        "xray_pdf_s3_key",
        "xray_pdf_s3_error",
    ):
        st.session_state.pop(key, None)

    st.info(
        "Upload a chest X-ray to display image measurements, "
        "model output, visual explanation and the PDF report."
    )

    st.stop()


# ============================================================
# SAVE IMAGE TO S3 AND RUN ANALYSIS
# ============================================================

uploaded_bytes = uploaded_file.getvalue()
image_key = hashlib.sha256(uploaded_bytes).hexdigest()

if st.session_state.get("xray_image_key") != image_key:
    st.session_state["xray_image_key"] = image_key
    st.session_state["xray_result"] = None

    st.session_state.pop("xray_error", None)
    st.session_state.pop("xray_traceback", None)
    st.session_state.pop("xray_s3_error", None)
    st.session_state.pop("xray_pdf_s3_error", None)

    st.session_state["xray_s3_key"] = None
    st.session_state["xray_pdf_s3_key"] = None

    # Upload the original image automatically.
    try:
        extension = uploaded_file.name.rsplit(".", 1)[-1].lower()

        content_types = {
            "png": "image/png",
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
        }

        object_key = upload_bytes(
            data=uploaded_bytes,
            filename=f"xray_{image_key[:16]}.{extension}",
            content_type=content_types[extension],
        )

        st.session_state["xray_s3_key"] = object_key

    except Exception as error:
        st.session_state["xray_s3_error"] = (
            f"{type(error).__name__}: {error}"
        )

    # Run the existing AI analysis workflow.
    try:
        with st.spinner(
            "Running the model, OpenCV enhancement and agent workflow..."
        ):
            pil_image = Image.open(
                io.BytesIO(uploaded_bytes)
            ).convert("RGB")

            original_rgb = np.asarray(pil_image)

            result = analyse_uploaded_xray(original_rgb)

            st.session_state["xray_result"] = result

    except Exception as error:
        st.session_state["xray_error"] = str(error)
        st.session_state["xray_traceback"] = traceback.format_exc()


# Display original-image storage status only.
if st.session_state.get("xray_s3_key"):
    st.success("Original X-ray saved to private S3 storage.")
elif st.session_state.get("xray_s3_error"):
    st.warning(
        "The original X-ray was not confirmed as saved to S3. "
        "Analysis can still continue."
    )


# ============================================================
# ERROR HANDLING
# ============================================================

if st.session_state.get("xray_error"):
    st.error("The image analysis could not be completed.")
    st.code(st.session_state["xray_error"])

    with st.expander("Technical details"):
        st.code(st.session_state.get("xray_traceback", ""))

    st.stop()


result = st.session_state.get("xray_result")

if result is None:
    st.stop()


# ============================================================
# 1. AGENT WORKFLOW STATUS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-heading">1. Agent workflow status</div>',
    unsafe_allow_html=True,
)

decision = result["decision"]

action = decision.get("action", "Not provided by controller")
status = decision.get("status", "Not provided by controller")
reason = decision.get("reason", "No decision reason was returned.")

col_action, col_status = st.columns(2)

with col_action:
    st.markdown("**Selected action**")
    st.write(action)

with col_status:
    st.markdown("**Workflow status**")
    st.write(status)

st.markdown("**Controller explanation**")
st.write(reason)

if result["gradcam_error"]:
    st.warning(
        "Grad-CAM could not be generated. The main model output remains "
        "available, but the visual explanation and its overlay are omitted. "
        f"Technical detail: {result['gradcam_error']}"
    )
elif result["overlay_rgb"] is not None:
    st.success(
        "The model output and Grad-CAM visualization are available."
    )

st.caption(
    "The controller is a rule-based workflow component. Its action and "
    "status reflect the values returned by your current agent code; "
    "they do not establish medical validity."
)


# ============================================================
# 2. ORIGINAL AND ENHANCED IMAGES + MEASUREMENTS
# ============================================================

st.divider()

st.markdown(
    '<div class="section-heading">'
    '2. Image comparison and measurements'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Compare the original image with its OpenCV contrast-enhanced version."
)

original_column, enhanced_column = st.columns(2, gap="medium")

with original_column:
    st.markdown("**Original X-ray**")

    st.image(
        result["original_rgb"],
        use_container_width=True,
    )

    st.markdown(
        '<div class="measurement-heading">'
        'Original X-ray measurements'
        '</div>',
        unsafe_allow_html=True,
    )

    display_image_measurements(
        result["original_rgb"],
        file_size_bytes=len(uploaded_bytes),
    )

with enhanced_column:
    st.markdown("**OpenCV-enhanced X-ray**")

    st.image(
        result["enhanced_rgb"],
        use_container_width=True,
    )

    st.markdown(
        '<div class="measurement-heading">'
        'Enhanced X-ray measurements'
        '</div>',
        unsafe_allow_html=True,
    )

    display_image_measurements(
        result["enhanced_rgb"],
    )

st.caption(
    "The original file size applies only to the uploaded file. "
    "The enhanced image is measured in memory and does not have a separate "
    "file size unless it is explicitly encoded or saved."
)

st.info(
    "These measurements describe image dimensions and pixel intensity. "
    "They are not measurements of lung size, heart size, lesions, or other "
    "anatomical structures. Enhancement can change intensity and contrast "
    "measurements without adding medical information."
)

st.caption(
    "Enhancement changes image appearance. It does not add medical "
    "information or guarantee better model accuracy."
)


# ============================================================
# 3. GRAD-CAM VISUAL EXPLANATION
# ============================================================

st.divider()

st.markdown(
    '<div class="section-heading">3. Visual explanation</div>',
    unsafe_allow_html=True,
)

if result["overlay_rgb"] is not None:
    st.write(
        "This experimental Grad-CAM visualization highlights image "
        "regions associated with the selected model output."
    )

    st.image(
        result["overlay_rgb"],
        caption="Original X-ray with Grad-CAM overlay",
        use_container_width=True,
    )

    st.markdown("**Where is the strongest relative activation?**")
    st.write(result["influential_region"])

    st.info(
        "Coloured areas indicate relative model activation, not confirmed "
        "disease locations. Grad-CAM is not validated here for clinical "
        "decisions."
    )
else:
    st.info(
        "The Grad-CAM visualization is unavailable for this image. "
        "The model output can still be reviewed below."
    )


# ============================================================
# 4. TOP-RANKED MODEL FINDING
# ============================================================

st.divider()

st.markdown(
    '<div class="section-heading">'
    '4. Understanding the model output'
    '</div>',
    unsafe_allow_html=True,
)

explanation = result["explanation"]

st.markdown("### Top-ranked model finding")
st.markdown(
    f"**{explanation.get('display_name', 'Unavailable')}**"
)

summary = explanation.get("summary", "")
if summary:
    st.write(summary)

meaning = explanation.get("meaning", "")
if meaning:
    st.markdown("**What does this mean?**")
    st.write(meaning)

context = explanation.get("context", "")
if context:
    st.markdown("**Important context**")
    st.write(context)

st.caption(
    "This is the highest-ranked output among the model's supported labels. "
    "Its raw score is not a calibrated individual disease probability. "
    "The result does not establish or exclude a diagnosis."
)


# ============================================================
# 5. GUIDANCE AND SAFETY
# ============================================================

st.divider()

st.markdown(
    '<div class="section-heading">5. Suggested next steps</div>',
    unsafe_allow_html=True,
)

st.info(
    explanation.get(
        "clinic_guidance",
        "Ask a qualified healthcare professional to interpret the original "
        "X-ray alongside your symptoms and medical history.",
    )
)

st.warning(
    explanation.get(
        "emergency_warning",
        "For severe difficulty breathing, severe chest pain, blue or grey "
        "lips, or new confusion, seek emergency medical care immediately. "
        "Do not wait for this AI result.",
    )
)

st.caption(
    "This application cannot determine whether a clinic visit is necessary "
    "or how urgently care is needed. Do not delay care because of this output."
)

with st.expander("Limitations and safety information"):
    disclaimer = explanation.get("disclaimer", "")

    if disclaimer:
        st.write(disclaimer)

    st.write(
        "A qualified healthcare professional should interpret the original "
        "X-ray alongside symptoms, examination findings and medical history. "
        "Grad-CAM is an exploratory interpretation method, not a validated "
        "disease-localization tool."
    )


# ============================================================
# 6. PDF REPORT AND AUTOMATIC S3 STORAGE
# ============================================================

st.divider()

st.markdown(
    '<div class="section-heading">6. Download your report</div>',
    unsafe_allow_html=True,
)

st.write(
    "Generate a PDF containing the original and enhanced images, available "
    "Grad-CAM visualization, model explanation and safety guidance."
)

pdf_explanation = dict(explanation)

existing_context = str(
    explanation.get("context", "")
).strip()

if result["overlay_rgb"] is not None:
    region_context = (
        "Grad-CAM region summary:\n"
        + result["influential_region"]
    )
else:
    region_context = (
        "Grad-CAM visualization was unavailable for this analysis."
    )

workflow_context = (
    f"Agent action: {action}\n"
    f"Workflow status: {status}\n"
    f"Controller reason: {reason}"
)

# Include the image measurements in the PDF explanation context too.
original_measurement_lines = calculate_image_measurements(
    result["original_rgb"],
    file_size_bytes=len(uploaded_bytes),
)

enhanced_measurement_lines = calculate_image_measurements(
    result["enhanced_rgb"],
)

measurement_context = "ORIGINAL X-RAY MEASUREMENTS:\n"
measurement_context += "\n".join(
    f"{label}: {value}"
    for label, value in original_measurement_lines
)

measurement_context += "\n\nENHANCED X-RAY MEASUREMENTS:\n"
measurement_context += "\n".join(
    f"{label}: {value}"
    for label, value in enhanced_measurement_lines
)

additional_context = (
    measurement_context
    + "\n\n"
    + region_context
    + "\n\n"
    + workflow_context
)

if existing_context:
    pdf_explanation["context"] = (
        existing_context + "\n\n" + additional_context
    )
else:
    pdf_explanation["context"] = additional_context

# If Grad-CAM is unavailable, use the original image as a placeholder.
# The report context explicitly states that Grad-CAM was unavailable.
pdf_overlay = (
    result["overlay_rgb"]
    if result["overlay_rgb"] is not None
    else result["original_rgb"]
)

try:
    with st.spinner("Preparing your PDF report..."):
        pdf_bytes = build_visual_pdf(
            original_rgb=result["original_rgb"],
            enhanced_rgb=result["enhanced_rgb"],
            overlay_rgb=pdf_overlay,
            explanation=pdf_explanation,
            selected_label=explanation.get("display_name"),
        )

    if (
        not isinstance(pdf_bytes, bytes)
        or not pdf_bytes.startswith(b"%PDF")
    ):
        st.error(
            "The PDF generator did not return a valid PDF file."
        )

    else:
        # Automatically upload the PDF once per uploaded image in this
        # Streamlit session. The S3 object remains private.
        if st.session_state.get("xray_pdf_s3_key") is None:
            try:
                pdf_s3_key = upload_bytes(
                    data=pdf_bytes,
                    filename=f"xrayguide_report_{image_key[:16]}.pdf",
                    content_type="application/pdf",
                )

                st.session_state["xray_pdf_s3_key"] = pdf_s3_key
                st.session_state.pop("xray_pdf_s3_error", None)

            except Exception as error:
                st.session_state["xray_pdf_s3_error"] = (
                    f"{type(error).__name__}: {error}"
                )

        # Intentionally do not display a PDF-upload success message.
        if (
            st.session_state.get("xray_pdf_s3_key") is None
            and st.session_state.get("xray_pdf_s3_error")
        ):
            st.warning(
                "The PDF is available to download, but its upload to "
                "S3 was not confirmed."
            )

        st.download_button(
            label="Download complete XrayGuide AI report (PDF)",
            data=pdf_bytes,
            file_name="xrayguide_ai_report.pdf",
            mime="application/pdf",
            type="primary",
            use_container_width=True,
            key="download_xrayguide_report",
        )

except Exception as error:
    st.error("The PDF could not be generated.")

    with st.expander("PDF generation error"):
        st.code(f"{type(error).__name__}: {error}")


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "XrayGuide AI | Experimental research prototype | "
    "Not for clinical diagnosis or medical triage"
)
