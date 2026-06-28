# 🖥️ Dashboard — Plant Disease Inference Demo

A Streamlit app that loads the trained models from this project and runs **live, side-by-side inference** on any leaf image you upload.

## Preview

> Upload a leaf photo → get predictions + confidence scores from every available model → see which one is most confident and why.

## Quick Start

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (default `http://localhost:8501`).

## Required Files

The app looks for trained model checkpoints in `../models/`:

| Model | Expected filename |
|---|---|
| MobileNetV2 ⭐ | `mobilenet_best.keras` |
| VGG16 | `20epochvgg16model.keras` |
| ResNet50 | `100epochrenet.keras` |
| EfficientNetB0 | `30eppochefficientnet.keras` |

You don't need all four — the dashboard automatically skips any model whose file isn't present, with a message explaining what's missing. See [`../models/README.md`](../models/README.md) for how to obtain these files (they're not committed to the repo due to size).

## Customizing

- **Different filenames/paths?** Edit `MODEL_CONFIG` at the top of `app.py`.
- **Class labels** live in `class_names.py`, sorted alphabetically to match the index order Keras assigned during training (`flow_from_directory` sorts folder names alphabetically — this is a common gotcha if you ever see scrambled predictions).
- **Deploying publicly?** This app works as-is on [Streamlit Community Cloud](https://streamlit.io/cloud) or [Hugging Face Spaces](https://huggingface.co/spaces) — just make sure your model files are downloadable at runtime (see the Hugging Face Hub option in the root README) since most free hosting tiers won't let you commit large `.keras` files directly.
