"""
Crop Disease Detection — Streamlit Application

Upload a leaf image and receive a predicted disease class with
confidence score.  Requires a trained model (best_model.h5) and
class mapping (class_mapping.json) in the artifacts directory.
"""

import json
import os
import sys

import streamlit as st
from PIL import Image

from src.predict import load_trained_model, preprocess_image, predict_image

# ---------------------------------------------------------------------------
# Configuration — override via environment variable if needed
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
MODEL_PATH    = os.path.join(ARTIFACTS_DIR, "best_model.h5")
CLASSES_PATH  = os.path.join(ARTIFACTS_DIR, "class_mapping.json")

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Crop Disease Detection",
    page_icon="🌿",
    layout="centered",
)

# ---------------------------------------------------------------------------
# Custom CSS for a polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main { max-width: 720px; margin: auto; }
    .stProgress > div > div > div { background: linear-gradient(90deg, #34d399, #10b981); }
    div[data-testid="stFileUploader"] { border: 2px dashed #6ee7b7; border-radius: 12px; padding: 1rem; }
    .disclaimer { background: #fef3c7; border-left: 4px solid #f59e0b;
                   padding: 0.75rem 1rem; border-radius: 6px; font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🌿 Crop Disease Detection")
st.markdown(
    "Upload a photograph of a plant leaf and the model will predict the most "
    "likely disease (or healthy status) along with a confidence score."
)

# ---------------------------------------------------------------------------
# Load model & class mapping (cached — loaded only once)
# ---------------------------------------------------------------------------
@st.cache_resource
def _load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return load_trained_model(MODEL_PATH)


@st.cache_data
def _load_classes():
    if not os.path.exists(CLASSES_PATH):
        return None
    with open(CLASSES_PATH, "r") as f:
        data = json.load(f)
    if isinstance(data, dict) and "classes" in data:
        return data["classes"]
    if isinstance(data, list):
        return data
    return None


model   = _load_model()
classes = _load_classes()

# ---------------------------------------------------------------------------
# Guard: missing artifacts
# ---------------------------------------------------------------------------
if model is None or classes is None:
    st.warning("⚠️ Trained model or class mapping not found.")
    st.info(
        f"Expected files:\n"
        f"- `{MODEL_PATH}`\n"
        f"- `{CLASSES_PATH}`\n\n"
        "Train the model using the Google Colab notebook, then copy "
        "`best_model.h5` and `class_mapping.json` from your Google Drive "
        f"into the local `{ARTIFACTS_DIR}/` folder.  See the README for details."
    )
    st.stop()

# Validate that model output dimension matches class count
expected_outputs = model.output_shape[-1]
if expected_outputs != len(classes):
    st.error(
        f"Model has {expected_outputs} outputs but class mapping has "
        f"{len(classes)} classes.  The artifacts are incompatible."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Image upload & prediction
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a leaf image",
    type=["jpg", "jpeg", "png"],
    help="Supported formats: JPG, JPEG, PNG",
)

if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)
    except Exception:
        st.error("The uploaded file could not be read as an image.")
        st.stop()

    if st.button("🔍 Predict", use_container_width=True):
        with st.spinner("Analysing image…"):
            try:
                img_array = preprocess_image(uploaded_file)
                results = predict_image(model, img_array, classes)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
                st.stop()

        top = results[0]

        st.success("Prediction complete!")
        st.markdown(f"### Predicted: **{top['class']}**")
        st.progress(top["confidence"])
        st.markdown(f"**Confidence:** {top['confidence'] * 100:.1f}%")

        if len(results) >= 3:
            with st.expander("Top 3 Predictions"):
                for i, r in enumerate(results[:3], 1):
                    st.write(f"{i}. **{r['class']}** — {r['confidence'] * 100:.1f}%")

    # Disclaimer
    st.markdown(
        '<div class="disclaimer">'
        "⚠️ <strong>Disclaimer:</strong> Predictions are model estimates based "
        "on the training dataset and are <em>not</em> guaranteed professional "
        "diagnoses. Consult an expert for actionable decisions."
        "</div>",
        unsafe_allow_html=True,
    )
