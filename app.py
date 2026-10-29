import hashlib
import html
import io
import traceback

import cv2
import numpy as np
import streamlit as st
from PIL import Image

from config.settings import AUTO_S3_SYNC_DEFAULT
from src.agent import analyze_xray
from src.pdf_report import build_visual_pdf
from s3_storage import get_s3_bucket, get_s3_region, list_files, upload_bytes


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="XrayGuide AI",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="expanded",
)

NAV_ITEMS = ["Home", "Upload Image", "Analysis", "Report", "Settings"]
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.session_state.setdefault("nav", "Home")
st.session_state.setdefault("score_threshold", 0.3)
st.session_state.setdefault("auto_s3_sync", AUTO_S3_SYNC_DEFAULT)


# ============================================================
# SMALL HTML / SVG HELPERS
# ============================================================

def html_block(text):
    """Collapse indented HTML so Markdown never treats it as a code block."""
    return " ".join(line.strip() for line in text.strip().splitlines())


def go_to(page):
    """Button callback: switch the sidebar navigation."""
    st.session_state["nav"] = page


def retry_analysis():
    """Clear the last failure while retaining the current upload and S3 object."""
    for key in ("xray_error", "xray_traceback"):
        st.session_state.pop(key, None)


def keyed_container(key, **kwargs):
    """st.container(key=...) where supported, plain container otherwise."""
    try:
        return st.container(key=key, **kwargs)
    except TypeError:
        return st.container(**kwargs)


def page_header(title, subtitle=""):
    sub = f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    st.markdown(f'<div class="main-title">{title}</div>{sub}', unsafe_allow_html=True)


def img_label(text):
    st.markdown(f'<div class="img-label">{text}</div>', unsafe_allow_html=True)


def html_table(headers, rows):
    head = "".join(f"<th>{h}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows
    )
    return (
        '<div class="xg-table-wrap"><table class="xg-table">'
        f"<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    )


LUNGS_SVG = (
    '<svg width="{s}" height="{s}" viewBox="0 0 48 48" fill="none" stroke="#22D3EE" '
    'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M24 4V22M24 22L19 28M24 22L29 28"/>'
    '<path d="M20 9C12 11 6 22 6 34C6 40 11 42 15 39C18 37 20 33 20 28Z" fill="rgba(34,211,238,.18)"/>'
    '<path d="M28 9C36 11 42 22 42 34C42 40 37 42 33 39C30 37 28 33 28 28Z" fill="rgba(34,211,238,.18)"/>'
    "</svg>"
)

CLOUD_UPLOAD_SVG = (
    '<svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="#1E6FE8" '
    'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M7 18a4 4 0 0 1-.5-7.97A6 6 0 0 1 18 9a4.5 4.5 0 0 1 0 9"/>'
    '<path d="M12 12v8M9 15l3-3 3 3"/></svg>'
)

S3_SVG = (
    '<svg width="42" height="42" viewBox="0 0 24 24">'
    '<path d="M12 3l8 3-8 3-8-3z" fill="#2F80ED"/>'
    '<path d="M4 6v12l8 3V9z" fill="#1E5FC2"/>'
    '<path d="M20 6v12l-8 3V9z" fill="#4A9BFF"/></svg>'
)

FILE_IMG_SVG = (
    '<svg width="16" height="16" viewBox="0 0 24 24"><rect x="2" y="2" width="20" height="20" rx="4" fill="#1E6FE8"/>'
    '<circle cx="9" cy="9" r="2" fill="#fff"/><path d="M4 19l5-6 4 4 3-3 4 5z" fill="#fff"/></svg>'
)

FILE_PDF_SVG = (
    '<svg width="16" height="16" viewBox="0 0 24 24"><rect x="3" y="2" width="18" height="20" rx="3" fill="#E5484D"/>'
    '<path d="M8 9h8M8 13h8M8 17h5" stroke="#fff" stroke-width="2" stroke-linecap="round"/></svg>'
)

HERO_ART_SVG = """
<svg class="hero-art" viewBox="0 0 300 380" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <radialGradient id="heat" cx="50%" cy="50%" r="50%">
      <stop offset="0" stop-color="#FF2D1A" stop-opacity=".95"/>
      <stop offset=".35" stop-color="#FFB300" stop-opacity=".85"/>
      <stop offset=".68" stop-color="#22D3A0" stop-opacity=".35"/>
      <stop offset="1" stop-color="#22D3A0" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="lung" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#7FB6E8" stop-opacity=".55"/>
      <stop offset="1" stop-color="#2D5F9A" stop-opacity=".22"/>
    </linearGradient>
  </defs>
  <g stroke="#7FB6E8" stroke-opacity=".28" stroke-width="3" fill="none" stroke-linecap="round">
    <path d="M70 120Q150 150 230 120"/><path d="M62 160Q150 195 238 160"/>
    <path d="M58 200Q150 238 242 200"/><path d="M58 240Q150 278 242 240"/>
    <path d="M150 20V350" stroke-dasharray="2 9" stroke-opacity=".4"/>
  </g>
  <path d="M135 60C95 80 60 150 55 240C52 300 75 330 110 322C135 316 140 290 140 250L140 90Z"
        fill="url(#lung)" stroke="#7FB6E8" stroke-opacity=".55" stroke-width="2"/>
  <path d="M165 60C205 80 240 150 245 240C248 300 225 330 190 322C165 316 160 290 160 250L160 90Z"
        fill="url(#lung)" stroke="#7FB6E8" stroke-opacity=".55" stroke-width="2"/>
  <path d="M150 18V92M150 92L132 125M150 92L168 125" stroke="#9CC9F0" stroke-opacity=".6"
        stroke-width="4" fill="none" stroke-linecap="round"/>
  <ellipse cx="188" cy="232" rx="62" ry="66" fill="url(#heat)"/>
</svg>
"""


# ============================================================
# GLOBAL STYLES
# ============================================================

def _svg_uri(body):
    svg = (
        "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' "
        "stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>" + body + "</svg>"
    )
    return "data:image/svg+xml;utf8," + svg.replace("<", "%3C").replace(">", "%3E").replace("#", "%23")


NAV_ICON_BODIES = [
    "<path d='M3 11l9-8 9 8'/><path d='M5 10v10h14V10'/>",                                   # Home
    "<path d='M7 18a4 4 0 0 1-.5-7.97A6 6 0 0 1 18 9a4.5 4.5 0 0 1 0 9'/><path d='M12 12v8M9 15l3-3 3 3'/>",  # Upload
    "<rect x='3' y='12' width='4' height='8'/><rect x='10' y='4' width='4' height='16'/><rect x='17' y='9' width='4' height='11'/>",  # Analysis
    "<path d='M14 3H6v18h12V7z'/><path d='M14 3v4h4M9 12h6M9 16h6'/>",                       # Report
    "<circle cx='12' cy='12' r='3'/><path d='M12 2v3M12 19v3M2 12h3M19 12h3M4.9 4.9L7 7M17 17l2.1 2.1M4.9 19.1L7 17M17 7l2.1-2.1'/>",  # Settings
]

NAV_ICON_CSS = "".join(
    f"""section[data-testid="stSidebar"] div[role="radiogroup"] > label:nth-of-type({i}) p::before
        {{background-image:url("{_svg_uri(body)}");}}"""
    for i, body in enumerate(NAV_ICON_BODIES, start=1)
)

BASE_CSS = """
<style>
/* ---------- chrome ---------- */
#MainMenu, footer, .stDeployButton, [data-testid="stToolbar"] {visibility: hidden; height: 0;}
header[data-testid="stHeader"] {background: transparent;}
.stApp {background: #FFFFFF; color: #0F1B2D;}
.block-container {max-width: 1200px; padding-top: 2rem; padding-bottom: 3rem;}

/* ---------- typography ---------- */
.main-title {font-size: 1.55rem; font-weight: 700; color: #0F1B2D; margin-bottom: .2rem;}
.subtitle {color: #64748B; font-size: .9rem; margin-bottom: 1.3rem;}
.xg-h3 {font-size: 1rem; font-weight: 650; color: #0F1B2D; margin: 1.3rem 0 .6rem;}
.img-label {text-align: center; font-size: .8rem; font-weight: 600; color: #475569; margin: .2rem 0 .4rem;}
div[data-testid="stImage"] img {border-radius: 8px;}

/* ---------- sidebar ---------- */
section[data-testid="stSidebar"] {background: #0F1B2D; border-right: none;}
section[data-testid="stSidebar"] > div {background: #0F1B2D;}
section[data-testid="stSidebar"] * {color: #CBD5E1;}
section[data-testid="stSidebar"] .xg-brand {display: flex; align-items: center; gap: .55rem;
    font-size: 1.1rem; font-weight: 700; padding: .2rem .4rem 1.2rem;}
section[data-testid="stSidebar"] .xg-brand span {color: #FFFFFF;}
section[data-testid="stSidebar"] .xg-brand b {color: #38BDF8;}
section[data-testid="stSidebar"] .xg-side-foot {font-size: .72rem; color: #64748B; padding: 1.5rem .4rem 0;}
section[data-testid="stSidebar"] div[role="radiogroup"] {gap: .25rem;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label {
    padding: .6rem .8rem; border-radius: 8px; width: 100%; cursor: pointer; align-items: center;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child {display: none;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover {background: rgba(255,255,255,.06);}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) {background: #1C3355;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) * {color: #FFFFFF;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p {
    display: flex; align-items: center; gap: .7rem; font-size: .92rem; margin: 0;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label p::before {
    content: ""; width: 17px; height: 17px; flex-shrink: 0; opacity: .85;
    background-repeat: no-repeat; background-size: contain; background-position: center;}

/* ---------- buttons ---------- */
button[kind="primary"], button[data-testid="stBaseButton-primary"],
div[data-testid="stDownloadButton"] button,
section[data-testid="stFileUploaderDropzone"] button {
    background: #1E6FE8 !important; color: #FFFFFF !important; border: none !important;
    border-radius: 6px !important; padding: .5rem 1.5rem !important; font-weight: 600 !important;}
button[kind="primary"] p, button[data-testid="stBaseButton-primary"] p,
div[data-testid="stDownloadButton"] button p,
section[data-testid="stFileUploaderDropzone"] button p {color: #FFFFFF !important;}
button[kind="primary"]:hover, button[data-testid="stBaseButton-primary"]:hover,
div[data-testid="stDownloadButton"] button:hover,
section[data-testid="stFileUploaderDropzone"] button:hover {background: #1A5FC9 !important;}

/* ---------- tabs ---------- */
button[data-baseweb="tab"] {font-weight: 600; font-size: .92rem;}
button[data-baseweb="tab"][aria-selected="true"] {color: #1E6FE8;}
div[data-baseweb="tab-highlight"] {background: #1E6FE8 !important;}

/* ---------- home hero ---------- */
.st-key-hero {background: radial-gradient(circle at 80% 40%, #12335C 0%, #0B1626 55%, #09111F 100%);
    border-radius: 14px; padding: 3.2rem 3rem; min-height: 430px; position: relative; overflow: hidden;}
.st-key-hero [data-testid="stVerticalBlock"] {position: relative; z-index: 1;}
.hero-art {position: absolute; right: 1rem; top: -1.5rem; height: 115%; width: auto; z-index: 0;
    pointer-events: none;}
.hero-text {max-width: 520px;}
.hero-head {display: flex; align-items: center; gap: 1rem; margin-bottom: .4rem;}
.hero-title {font-size: 3rem; font-weight: 800; color: #FFFFFF; line-height: 1.05;}
.hero-title b {color: #38BDF8;}
.hero-sub {font-size: 1.05rem; font-weight: 600; color: #FFFFFF; margin: .5rem 0 1.4rem;}
.hero-tags {color: #38BDF8; font-size: .95rem; margin-bottom: 1rem;}
.hero-tags i {font-style: normal; color: #38BDF8; margin: 0 .55rem;}
.hero-desc {color: #CBD5E1; font-size: .88rem; line-height: 1.6; margin-bottom: 1.6rem;}
.xg-disclaimer {color: #64748B; font-size: .78rem; margin-top: .9rem;}

/* ---------- upload ---------- */
.st-key-upload_card {border: 2px dashed #CBD5E1; border-radius: 12px; background: #FFFFFF;
    padding: 2.2rem 1rem 1.2rem; max-width: 660px;}
.st-key-upload_card [data-testid="stVerticalBlock"] {align-items: center; gap: .35rem;}
.up-icon {text-align: center;}
.up-text {text-align: center; font-size: .9rem; color: #334155;}
.up-or {text-align: center; font-size: .8rem; color: #94A3B8;}
.up-foot {text-align: center; font-size: .72rem; color: #64748B; margin-top: .6rem;}
section[data-testid="stFileUploaderDropzone"] {background: transparent; border: none; justify-content: center; padding: .4rem;}
section[data-testid="stFileUploaderDropzone"] [data-testid="stFileUploaderDropzoneInstructions"] {display: none;}

/* ---------- tables ---------- */
.xg-table-wrap {overflow-x: auto; margin-bottom: .5rem;}
.xg-table {width: 100%; border-collapse: separate; border-spacing: 0; border: 1px solid #E2E8F0;
    border-radius: 8px; overflow: hidden; font-size: .85rem;}
.xg-table th {background: #F8FAFC; color: #334155; font-weight: 650; text-align: left; padding: .6rem .9rem;}
.xg-table td {color: #0F1B2D; padding: .6rem .9rem; border-top: 1px solid #EEF1F6;}
.conf {display: flex; align-items: center; gap: .8rem;}
.conf span {min-width: 2.2rem;}
.bar {width: 110px; height: 8px; background: #E8EEF7; border-radius: 99px; overflow: hidden;}
.bar i {display: block; height: 100%; background: #1E6FE8; border-radius: 99px;}
.bar i.low {background: #BCD3F7;}
.ftype {display: inline-flex; align-items: center; gap: .5rem;}

/* ---------- badges & boxes ---------- */
.meta-row {display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .6rem;
    margin-bottom: .4rem;}
.badge-blue {background: #E8F1FE; color: #1E6FE8; font-size: .75rem; font-weight: 600; padding: .3rem .7rem;
    border-radius: 6px; border: 1px solid #C7DCFA;}
.meta-thr {font-size: .75rem; color: #475569; border: 1px solid #E2E8F0; border-radius: 6px; padding: .3rem .7rem;}
.title-row {display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: .6rem;}
.badge-green {background: #E7F8EE; color: #15803D; font-size: .75rem; font-weight: 600; padding: .3rem .8rem;
    border-radius: 99px; border: 1px solid #BBF7D0;}
.badge-red {background: #FDECEC; color: #B91C1C; font-size: .75rem; font-weight: 600; padding: .3rem .8rem;
    border-radius: 99px; border: 1px solid #FECACA;}
.xg-success {background: #ECFDF3; border: 1px solid #BBF7D0; color: #166534; padding: .95rem 1.1rem;
    border-radius: 8px; font-size: .9rem; margin-bottom: 1.1rem;}
.xg-note {background: #F1F5F9; border: 1px solid #E2E8F0; color: #334155; padding: .9rem 1.1rem;
    border-radius: 8px; font-size: .82rem; margin-top: 1.1rem;}
.xg-checks {list-style: none; padding: 0; margin: .3rem 0 0; font-size: .9rem; color: #334155;}
.xg-checks li {margin: .35rem 0;}
.xg-checks li::before {content: "\\2713"; color: #16A34A; font-weight: 700; margin-right: .65rem;}
.s3-card {display: flex; align-items: center; gap: 2.2rem; flex-wrap: wrap; border: 1px solid #E2E8F0;
    border-radius: 10px; padding: 1.1rem 1.4rem; margin: 1rem 0 .4rem;}
.s3-card .k {font-size: .72rem; color: #64748B;}
.s3-card .v {font-size: .92rem; font-weight: 600; color: #0F1B2D;}

/* ---------- mobile: keep image pairs side by side ---------- */
@media (max-width: 640px) {
    .block-container {padding-left: 1rem; padding-right: 1rem;}
    div[data-testid="stHorizontalBlock"] {flex-wrap: nowrap !important; gap: .6rem !important;}
    div[data-testid="stColumn"], div[data-testid="column"] {min-width: 0 !important; flex: 1 1 0 !important;}
    .hero-title {font-size: 2.1rem;}
    .hero-art {opacity: .25;}
    .st-key-hero {padding: 2rem 1.4rem;}
}
"""

st.markdown(BASE_CSS + NAV_ICON_CSS + "</style>", unsafe_allow_html=True)


# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

with st.sidebar:
    st.markdown(
        html_block(
            f'<div class="xg-brand">{LUNGS_SVG.format(s=26)}<span>XrayGuide <b>AI</b></span></div>'
        ),
        unsafe_allow_html=True,
    )
    selected_page = st.radio(
        "Navigation",
        NAV_ITEMS,
        key="nav",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="xg-side-foot">XrayGuide AI v1.2 • Research Preview</div>',
        unsafe_allow_html=True,
    )


# ============================================================
# IMAGE HELPERS
# ============================================================

def _format_size(size_bytes):
    if size_bytes is None:
        return "—"
    if size_bytes >= 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    return f"{size_bytes / 1024:.0f} KB"


def image_metrics(image_rgb, size_bytes=None):
    """Dimensions, mean intensity, normalised contrast and file size for one image."""
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    if image.ndim == 2:
        gray = image
    elif image.ndim == 3 and image.shape[2] == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        raise ValueError("Unsupported image format for measurement calculation.")

    height, width = gray.shape[:2]
    p5, p95 = np.percentile(gray, [5, 95])
    return {
        "Dimensions": f"{width} × {height}",
        "Mean Intensity": f"{float(np.mean(gray)):.1f}",
        "Contrast": f"{float(p95 - p5) / 255:.2f}",
        "File Size": _format_size(size_bytes),
    }


def encoded_size(image_rgb, extension):
    """File size of an image when encoded in the same format as the upload."""
    ext = ".png" if extension == "png" else ".jpg"
    image = np.asarray(image_rgb)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, buffer = cv2.imencode(ext, image)
    return int(buffer.nbytes) if ok else None


def comparison_table_html(result):
    original = image_metrics(result["original_rgb"], result.get("original_size"))
    enhanced = image_metrics(result["enhanced_rgb"], result.get("enhanced_size"))
    rows = [[html.escape(key), original[key], enhanced[key]] for key in original]
    return html_table(["Metric", "Original", "Enhanced"], rows)


def create_gradcam_images(original_rgb, heatmap):
    activation_map = np.asarray(heatmap, dtype=np.float32)

    if activation_map.ndim == 3:
        if activation_map.shape[2] == 1:
            activation_map = activation_map[:, :, 0]
        else:
            activation_map = cv2.cvtColor(activation_map, cv2.COLOR_RGB2GRAY)

    if activation_map.ndim != 2:
        raise ValueError("Grad-CAM must return a 2D activation map.")

    activation_map = np.nan_to_num(activation_map, nan=0.0, posinf=1.0, neginf=0.0)
    activation_map = cv2.normalize(activation_map, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    activation_map = cv2.resize(activation_map, (original_rgb.shape[1], original_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = np.clip(activation_map * 255, 0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay_rgb = cv2.addWeighted(original_rgb, 0.60, heatmap_rgb, 0.40, 0)
    return overlay_rgb, activation_map


def describe_influential_region(activation_map):
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
                region_means.append((float(np.mean(region)), row_names[row], column_names[column]))

    if not region_means:
        return "The influential image region could not be estimated."

    _, vertical, horizontal = max(region_means, key=lambda item: item[0])
    return (
        f"The highest mean relative activation in this coarse grid "
        f"appears in the {vertical}, {horizontal} part of the image "
        f"(viewer's perspective)."
    )


def analyse_uploaded_xray(original_rgb):
    agent_result = analyze_xray(original_rgb)

    if not isinstance(agent_result, dict):
        raise ValueError("The agent returned an unexpected result.")

    if agent_result.get("status") != "success":
        raise ValueError(agent_result.get("message", "The agent could not complete the analysis."))

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
        enhanced_rgb = cv2.cvtColor(enhanced_image, cv2.COLOR_GRAY2RGB)
    elif enhanced_image.ndim == 3 and enhanced_image.shape[2] == 3:
        enhanced_rgb = enhanced_image
    else:
        raise ValueError("The agent returned an unsupported enhanced image format.")

    overlay_rgb = None
    activation_map = None
    gradcam_error = agent_result.get("gradcam_error")
    influential_region = "The influential region could not be estimated."

    if isinstance(gradcam_result, dict):
        heatmap = gradcam_result.get("heatmap")
        if heatmap is not None:
            try:
                overlay_rgb, activation_map = create_gradcam_images(original_rgb, heatmap)
                influential_region = describe_influential_region(activation_map)
            except Exception as exc:
                gradcam_error = str(exc)

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
# S3 HELPERS
# ============================================================

def normalise_stored_file(item):
    """Accept dicts (type/name/date or S3-style Key/LastModified) or plain strings."""
    if isinstance(item, str):
        item = {"name": item}
    name = str(item.get("name") or item.get("key") or item.get("Key") or "—")
    date = item.get("date") or item.get("last_modified") or item.get("LastModified") or "—"
    if hasattr(date, "strftime"):
        date = date.strftime("%Y-%m-%d %H:%M")
    declared = str(item.get("type", "")).lower()
    if declared in ("image", "report"):
        kind = declared
    else:
        kind = "report" if name.lower().endswith(".pdf") else "image"
    return kind, name.rsplit("/", 1)[-1], str(date)


# ============================================================
# HOME PAGE
# ============================================================

if selected_page == "Home":
    hero = keyed_container("hero")
    with hero:
        st.markdown(
            html_block(
                f"""
                {HERO_ART_SVG}
                <div class="hero-text">
                  <div class="hero-head">{LUNGS_SVG.format(s=76)}
                    <div class="hero-title">XrayGuide <b>AI</b></div></div>
                  <div class="hero-sub">AI-Powered Chest X-ray Analysis</div>
                  <div class="hero-tags">Enhance<i>•</i>Analyze<i>•</i>Visualize<i>•</i>Report<i>•</i>Store</div>
                  <div class="hero-desc">A web application that combines OpenCV 5, AI, and AWS
                  to help you explore chest X-ray images with enhanced
                  visualization, AI-assisted findings, and downloadable reports.</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        st.button("Get Started", type="primary", on_click=go_to, args=("Upload Image",), key="get_started")

    st.markdown(
        '<div class="xg-disclaimer">Research and educational use only. This prototype is not a '
        "diagnostic or medical triage tool.</div>",
        unsafe_allow_html=True,
    )


# ============================================================
# UPLOAD IMAGE PAGE
# ============================================================

elif selected_page == "Upload Image":
    page_header("Upload Chest X-ray Image", "Choose a chest X-ray image for analysis")

    upload_card = keyed_container("upload_card")
    with upload_card:
        st.markdown(
            html_block(
                f"""
                <div class="up-icon">{CLOUD_UPLOAD_SVG}</div>
                <div class="up-text">Drag and drop your image here</div>
                <div class="up-or">or</div>
                """
            ),
            unsafe_allow_html=True,
        )
        uploaded_file = st.file_uploader(
            "Upload chest X-ray image",
            type=["png", "jpg", "jpeg"],
            label_visibility="collapsed",
        )
        st.markdown(
            '<div class="up-foot">Supported formats: JPG, PNG, JPEG (Max size: 10 MB)</div>',
            unsafe_allow_html=True,
        )

    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()

        if len(uploaded_bytes) > MAX_UPLOAD_BYTES:
            st.error("This file is larger than 10 MB. Choose a smaller image.")
        else:
            image_key = hashlib.sha256(uploaded_bytes).hexdigest()

            is_new_image = st.session_state.get("xray_image_key") != image_key
            if is_new_image:
                st.session_state["xray_image_key"] = image_key
                for stale_key in ("xray_result", "xray_error", "xray_traceback", "xray_s3_key", "xray_s3_error", "pdf_cache"):
                    st.session_state.pop(stale_key, None)

            should_analyze = (
                is_new_image
                or (
                    st.session_state.get("xray_result") is None
                    and not st.session_state.get("xray_error")
                )
            )
            if should_analyze:
                extension = uploaded_file.name.rsplit(".", 1)[-1].lower()
                if (
                    st.session_state.get("auto_s3_sync", True)
                    and not st.session_state.get("xray_s3_key")
                ):
                    try:
                        content_types = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg"}
                        object_key = upload_bytes(
                            data=uploaded_bytes,
                            filename=f"xray_{image_key[:16]}.{extension}",
                            content_type=content_types[extension],
                        )
                        st.session_state["xray_s3_key"] = object_key
                    except Exception as error:
                        st.session_state["xray_s3_error"] = str(error)

                try:
                    with st.spinner("Analyzing image…"):
                        pil_image = Image.open(io.BytesIO(uploaded_bytes)).convert("RGB")
                        original_rgb = np.asarray(pil_image)
                        result = analyse_uploaded_xray(original_rgb)
                        result["original_size"] = len(uploaded_bytes)
                        result["enhanced_size"] = encoded_size(result["enhanced_rgb"], extension)
                    st.session_state["xray_result"] = result
                except Exception as error:
                    st.session_state["xray_error"] = str(error)
                    st.session_state["xray_traceback"] = traceback.format_exc()

            if st.session_state.get("xray_error"):
                st.error(f"Analysis failed: {st.session_state['xray_error']}")
                st.button(
                    "Retry analysis",
                    type="primary",
                    on_click=retry_analysis,
                    key="retry_analysis",
                )
                with st.expander("Technical details"):
                    st.code(st.session_state.get("xray_traceback", ""))
            elif st.session_state.get("xray_result") is not None:
                st.markdown(
                    '<div class="xg-success">Image analyzed successfully.</div>',
                    unsafe_allow_html=True,
                )
                if st.session_state.get("xray_s3_error"):
                    st.warning("The image was analyzed, but it could not be saved to cloud storage.")
                elif not st.session_state.get("xray_s3_key"):
                    st.info("Automatic S3 image sync is off; this image was not uploaded.")
                st.button("View Analysis", type="primary", on_click=go_to, args=("Analysis",), key="view_analysis")


# ============================================================
# ANALYSIS PAGE (Enhancement / AI Findings / Grad-CAM / Image Quality)
# ============================================================

elif selected_page == "Analysis":
    result = st.session_state.get("xray_result")

    if result is None:
        page_header("Analysis", "Enhancement, AI-assisted findings, Grad-CAM and image quality")
        st.info("Upload a chest X-ray first to see the analysis.")
        st.button("Go to Upload Image", type="primary", on_click=go_to, args=("Upload Image",), key="go_upload")
    else:
        tab_enh, tab_find, tab_cam, tab_quality = st.tabs(
            ["Image Enhancement", "AI Findings", "Grad-CAM", "Image Quality"]
        )

        # ---------- Image Enhancement ----------
        with tab_enh:
            page_header("Image Enhancement", "Original vs Enhanced Image")
            col1, col2 = st.columns(2)
            with col1:
                img_label("Original")
                st.image(result["original_rgb"], use_container_width=True)
            with col2:
                img_label("Enhanced (OpenCV 5)")
                st.image(result["enhanced_rgb"], use_container_width=True)

            st.markdown('<div class="xg-h3">Image Quality Comparison</div>', unsafe_allow_html=True)
            st.markdown(comparison_table_html(result), unsafe_allow_html=True)

        # ---------- AI-Assisted Findings ----------
        with tab_find:
            page_header("AI Analysis & Findings")
            threshold = float(st.session_state.get("score_threshold", 0.3))
            st.markdown(
                f'<div class="meta-row"><span class="badge-blue">Model: TorchXRayVision (Pretrained)</span>'
                f'<span class="meta-thr">Raw score display threshold: {threshold:g}</span></div>',
                unsafe_allow_html=True,
            )
            st.markdown(
                '<div class="xg-h3">Top Model Outputs (raw scores, not probabilities)</div>',
                unsafe_allow_html=True,
            )

            predictions = result.get("predictions", {})
            ranked = sorted(predictions.items(), key=lambda item: float(item[1]), reverse=True)[:5]
            rows = []
            for rank, (finding, score) in enumerate(ranked, start=1):
                value = float(score)
                width = max(0.0, min(1.0, value)) * 100
                low = "" if value >= threshold else "low"
                rows.append(
                    [
                        str(rank),
                        html.escape(str(finding)),
                        f'<div class="conf"><span>{value:.2f}</span>'
                        f'<div class="bar"><i class="{low}" style="width:{width:.0f}%"></i></div></div>',
                    ]
                )
            st.markdown(html_table(["Rank", "Model label", "Raw model score"], rows), unsafe_allow_html=True)

            explanation = result.get("explanation", {})
            if explanation.get("display_name") or explanation.get("summary"):
                with st.expander(f"About the top finding: {explanation.get('display_name', '')}"):
                    st.write(explanation.get("summary"))

        # ---------- Grad-CAM ----------
        with tab_cam:
            page_header("Grad-CAM Visualization", "Shows regions that influenced the model’s prediction")
            if result.get("overlay_rgb") is None:
                message = result.get("gradcam_error") or "Grad-CAM could not be generated for this image."
                st.warning(f"Grad-CAM is unavailable: {message}")
            else:
                col1, col2 = st.columns(2)
                with col1:
                    img_label("Original Image")
                    st.image(result["original_rgb"], use_container_width=True)
                with col2:
                    img_label("Grad-CAM Heatmap")
                    st.image(result["overlay_rgb"], use_container_width=True)
                st.caption(result["influential_region"])

            st.markdown(
                '<div class="xg-note"><b>Note:</b> Grad-CAM visualizations are available where supported '
                "and may not be generated for all models or inputs.</div>",
                unsafe_allow_html=True,
            )

        # ---------- Image Quality ----------
        with tab_quality:
            page_header("Image Quality Analysis", "Detailed comparison of original and enhanced images")
            _, col1, col2, _ = st.columns([1, 2, 2, 1])
            with col1:
                img_label("Original")
                st.image(result["original_rgb"], use_container_width=True)
            with col2:
                img_label("Enhanced")
                st.image(result["enhanced_rgb"], use_container_width=True)

            st.markdown('<div class="xg-h3">&nbsp;</div>', unsafe_allow_html=True)
            st.markdown(comparison_table_html(result), unsafe_allow_html=True)


# ============================================================
# REPORT PAGE (PDF report + Cloud storage)
# ============================================================

elif selected_page == "Report":
    tab_report, tab_s3 = st.tabs(["Generate Report", "Cloud Storage"])

    # ---------- Generate Report ----------
    with tab_report:
        page_header("Generate Report")
        result = st.session_state.get("xray_result")

        if result is None:
            st.info("Complete an analysis first. Upload an image to generate a report.")
            st.button("Go to Upload Image", type="primary", on_click=go_to, args=("Upload Image",), key="go_upload_report")
        else:
            st.markdown(
                '<div class="xg-success">Your analysis is complete! You can download a detailed PDF '
                "report with all results, images, and findings. The PDF is not uploaded to S3.</div>",
                unsafe_allow_html=True,
            )

            explanation = result["explanation"]
            pdf_overlay = result["overlay_rgb"] if result["overlay_rgb"] is not None else result["original_rgb"]
            image_key = st.session_state.get("xray_image_key")

            try:
                cached = st.session_state.get("pdf_cache")
                if cached and cached[0] == image_key:
                    pdf_bytes = cached[1]
                else:
                    pdf_bytes = build_visual_pdf(
                        original_rgb=result["original_rgb"],
                        enhanced_rgb=result["enhanced_rgb"],
                        overlay_rgb=pdf_overlay,
                        explanation=dict(explanation),
                        selected_label=explanation.get("display_name"),
                    )
                    st.session_state["pdf_cache"] = (image_key, pdf_bytes)

                download_args = dict(
                    label="Download PDF Report",
                    data=pdf_bytes,
                    file_name="xrayguide_ai_report.pdf",
                    mime="application/pdf",
                    type="primary",
                )
                try:
                    st.download_button(icon=":material/download:", **download_args)
                except TypeError:  # older Streamlit without the icon argument
                    st.download_button(**download_args)
            except Exception as error:
                st.error(f"Could not generate the PDF: {error}")

            st.markdown(
                html_block(
                    """
                    <div class="xg-h3">Report Includes:</div>
                    <ul class="xg-checks">
                      <li>Original &amp; enhanced images</li>
                      <li>AI findings and raw model scores (not probabilities)</li>
                      <li>Grad-CAM visualizations (where available)</li>
                      <li>Image quality measurements</li>
                      <li>Limitations and disclaimer</li>
                    </ul>
                    """
                ),
                unsafe_allow_html=True,
            )

    # ---------- Cloud Storage ----------
    with tab_s3:
        bucket, region = "Not configured", get_s3_region()

        stored_files, storage_error = [], None
        try:
            bucket = get_s3_bucket()
            stored_files = list(list_files() or [])
        except Exception as error:
            storage_error = str(error)

        badge = (
            '<span class="badge-red">● Not connected</span>'
            if storage_error
            else '<span class="badge-green">● Connected</span>'
        )
        st.markdown(
            f'<div class="title-row"><div class="main-title">Cloud Storage (Amazon S3)</div>{badge}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            html_block(
                f"""
                <div class="s3-card">
                  {S3_SVG}
                  <div><div class="k">Bucket Name</div><div class="v">{html.escape(bucket)}</div></div>
                  <div><div class="k">Region</div><div class="v">{html.escape(region)}</div></div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        st.markdown('<div class="xg-h3">Stored Files</div>', unsafe_allow_html=True)
        if storage_error:
            st.warning(f"Could not read the bucket: {storage_error}")
        elif not stored_files:
            st.info("No files stored yet. Uploaded images will appear here.")
        else:
            rows = []
            for item in stored_files:
                kind, name, date = normalise_stored_file(item)
                icon = FILE_PDF_SVG if kind == "report" else FILE_IMG_SVG
                rows.append(
                    [
                        f'<span class="ftype">{icon}{kind.capitalize()}</span>',
                        html.escape(name),
                        html.escape(date),
                    ]
                )
            st.markdown(html_table(["Type", "Name", "Date"], rows), unsafe_allow_html=True)


# ============================================================
# SETTINGS PAGE
# ============================================================

elif selected_page == "Settings":
    page_header("Settings", "Configure display and cloud-storage preferences")

    def _sync_score_threshold():
        st.session_state["score_threshold"] = st.session_state["_score_widget"]

    st.slider(
        "Raw score display threshold",
        0.0,
        1.0,
        value=float(st.session_state.get("score_threshold", 0.3)),
        step=0.05,
        key="_score_widget",
        on_change=_sync_score_threshold,
    )
    st.caption("The threshold only changes score-bar styling; it does not filter results.")
    st.checkbox(
        "Automatically sync uploaded images to private S3 storage",
        key="auto_s3_sync",
        help="When enabled, newly uploaded images are sent to the configured private S3 bucket.",
    )
    st.caption("The model architecture is fixed to the bundled TorchXRayVision DenseNet-121.")