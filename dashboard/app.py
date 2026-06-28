"""
Plant Disease Classification — Multi-Model Inference Dashboard
================================================================
Upload a leaf image and compare predictions across all 4 trained CNNs
(ResNet50, VGG16, MobileNetV2, EfficientNetB0) side by side.

Run with:
    streamlit run app.py

Place your trained .keras files in `../models/` using the filenames
below (or edit MODEL_CONFIG to point elsewhere). Models that aren't
found are skipped gracefully so the app still runs with whichever
checkpoints you have on disk.
"""

import os

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model

from class_names import CLASS_NAMES, format_class_name

# ----------------------------------------------------------------------
# Config — edit paths/filenames here if your checkpoints differ
# ----------------------------------------------------------------------
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_CONFIG = {
    "MobileNetV2": {
        "path": os.path.join(MODELS_DIR, "mobilenet_best.keras"),
        "target_size": (224, 224),
        "recommended": True,
    },
    "VGG16": {
        "path": os.path.join(MODELS_DIR, "20epochvgg16model.keras"),
        "target_size": (224, 224),
        "recommended": False,
    },
    "ResNet50": {
        "path": os.path.join(MODELS_DIR, "100epochrenet.keras"),
        "target_size": (256, 256),
        "recommended": False,
    },
    "EfficientNetB0": {
        "path": os.path.join(MODELS_DIR, "30eppochefficientnet.keras"),
        "target_size": (224, 224),
        "recommended": False,
    },
}

st.set_page_config(
    page_title="Plant Disease Classifier",
    page_icon="🌿",
    layout="wide",
)


# ----------------------------------------------------------------------
# Cached model loading — Streamlit reruns the script on every interaction,
# so caching keeps us from reloading multi-hundred-MB models each time.
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_available_models():
    loaded = {}
    missing = []
    for name, cfg in MODEL_CONFIG.items():
        if os.path.exists(cfg["path"]):
            try:
                loaded[name] = load_model(cfg["path"], compile=False)
            except Exception as e:  # noqa: BLE001
                missing.append((name, f"failed to load: {e}"))
        else:
            missing.append((name, "file not found"))
    return loaded, missing


def predict_single_image(model, pil_img: Image.Image, target_size):
    img = pil_img.convert("RGB").resize(target_size)
    arr = np.asarray(img, dtype="float32") / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    top3_idx = preds.argsort()[-3:][::-1]

    return {
        "predicted_class": CLASS_NAMES[top_idx],
        "confidence": float(preds[top_idx] * 100),
        "top3": [(CLASS_NAMES[i], float(preds[i] * 100)) for i in top3_idx],
    }


# ----------------------------------------------------------------------
# Sidebar — project context
# ----------------------------------------------------------------------
with st.sidebar:
    st.title("🌿 About this Project")
    st.markdown(
        """
        Comparative CNN-based plant leaf disease classification on the
        **PlantVillage** dataset (39 classes, ~54k images), built as an
        M.Tech thesis project at **JNTUH**.

        **Models compared:**
        - ResNet50
        - VGG16
        - MobileNetV2 ⭐ *(recommended — best accuracy/size tradeoff)*
        - EfficientNetB0

        [📂 View on GitHub](#) · [📄 Read the full write-up](../README.md)
        """
    )
    st.divider()
    st.caption("Author: Galib Amin · Supervisor: Dr. Athota Kavitha")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
st.title("🌿 Plant Disease Classification Dashboard")
st.write(
    "Upload a photo of a plant leaf and compare live predictions from "
    "every trained model in this project."
)

models, missing = load_available_models()

if missing:
    with st.expander(f"⚠️ {len(missing)} model(s) not loaded — click for details"):
        for name, reason in missing:
            st.write(f"- **{name}**: {reason}")
        st.info(
            "Place the corresponding `.keras` file in the `models/` folder "
            "(see `models/README.md`) to enable it here."
        )

if not models:
    st.error(
        "No trained models were found. Add at least one `.keras` file to "
        "the `models/` folder to use this dashboard."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a leaf image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    col_img, col_results = st.columns([1, 2])

    with col_img:
        st.image(image, caption="Uploaded image", use_container_width=True)

    results = []
    top3_rows = []

    with st.spinner("Running inference across all available models..."):
        for name, model in models.items():
            cfg = MODEL_CONFIG[name]
            out = predict_single_image(model, image, cfg["target_size"])
            results.append(
                {
                    "Model": name + (" ⭐" if cfg["recommended"] else ""),
                    "Predicted Class": format_class_name(out["predicted_class"]),
                    "Confidence (%)": round(out["confidence"], 2),
                }
            )
            for rank, (cls, conf) in enumerate(out["top3"], start=1):
                top3_rows.append(
                    {
                        "Model": name,
                        "Rank": rank,
                        "Predicted Class": format_class_name(cls),
                        "Confidence (%)": round(conf, 2),
                    }
                )

    results_df = pd.DataFrame(results)
    top3_df = pd.DataFrame(top3_rows)

    with col_results:
        st.subheader("Predictions by model")
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        best = results_df.loc[results_df["Confidence (%)"].idxmax()]
        st.success(
            f"**Most confident model:** {best['Model']} predicted "
            f"**{best['Predicted Class']}** with **{best['Confidence (%)']}%** confidence."
        )

        st.bar_chart(results_df.set_index("Model")["Confidence (%)"])

    with st.expander("See top-3 predictions per model"):
        st.dataframe(top3_df, use_container_width=True, hide_index=True)
else:
    st.info("👆 Upload an image to get started.")
