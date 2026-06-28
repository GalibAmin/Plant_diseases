# 🌿 Plant Disease Classification using Transfer Learning (CNNs)

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A comparative study of four CNN architectures — **ResNet50, VGG16, MobileNetV2, and EfficientNetB0** — for multi-class plant leaf disease classification on the **PlantVillage dataset** (39 classes, ~54,300 images), built as part of an M.Tech thesis at JNTUH.

> This repository documents the **complete workflow**: training on Kaggle (GPU), evaluation/EDA on Google Colab, real-world image testing, and a deployable inference dashboard.

---

## 📑 Table of Contents
- [Overview](#-overview)
- [Dataset](#-dataset)
- [Workflow](#-workflow)
- [Models Compared](#-models-compared)
- [Results](#-results)
- [Repository Structure](#-repository-structure)
- [Setup & Installation](#-setup--installation)
- [Running the Notebooks](#-running-the-notebooks)
- [Running the Dashboard](#-running-the-dashboard)
- [Model Weights](#-model-weights)
- [Tech Stack](#-tech-stack)
- [Future Work](#-future-work)
- [Author & Acknowledgements](#-author--acknowledgements)

---

## 📌 Overview

Plant diseases cause significant crop losses worldwide, and early, accurate diagnosis is critical for farmers and agricultural systems. This project benchmarks four widely-used CNN architectures using **transfer learning** (ImageNet-pretrained, frozen convolutional base + custom classification head) to identify 39 plant leaf classes (healthy + diseased) across 14 plant species.

The goal was to determine **not just the most accurate model, but the most deployment-practical one** — balancing accuracy, model size, and inference speed for real-world / mobile use cases.

This was completed as an M.Tech (Integrated Dual Program) thesis project at **JNTUH University College of Engineering, Science and Technology, Hyderabad**, under the supervision of **Dr. Athota Kavitha**.

## 📊 Dataset

**[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)** (unaugmented version)

| Metric | Value |
|---|---|
| Total images | ~54,306 |
| Total classes | 39 |
| Unique plant species | 14 |
| Healthy classes | 12 |
| Diseased classes | 26 |
| Other | 1 (`Background_without_leaves`) |
| Train/Validation split | 80% / 20% |

**Species covered:** Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Bell Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.

## 🔁 Workflow

This project intentionally used a **split-compute pipeline** to make the most of free-tier resources:

```
┌─────────────────────┐     ┌──────────────────────┐     ┌────────────────────────┐
│   Kaggle Notebooks   │ ──▶ │   Google Drive        │ ──▶ │   Google Colab          │
│   (Free GPU - train  │     │   (store .keras        │     │   (load models, EDA,   │
│    all 4 models)      │     │    model checkpoints)  │     │    evaluation, testing)│
└─────────────────────┘     └──────────────────────┘     └────────────────────────┘
                                                                       │
                                                                       ▼
                                                          ┌────────────────────────┐
                                                          │   Streamlit Dashboard   │
                                                          │   (local inference demo)│
                                                          └────────────────────────┘
```

1. **Training (`notebooks/01_kaggle_training.ipynb`)** — All 4 models trained on Kaggle's free GPU, using `ImageDataGenerator` (rescale + 80/20 split), `EarlyStopping`, `ReduceLROnPlateau`, and `ModelCheckpoint`. Trained weights exported as `.keras` files.
2. **EDA & Comparison (`notebooks/02_eda_and_model_comparison.ipynb`)** — Dataset exploration (class distribution, healthy vs. diseased split), model checkpoints reloaded from Google Drive, evaluated side-by-side.
3. **Real Image Testing (`notebooks/03_real_image_inference_test.ipynb`)** — All 4 trained models tested on an unseen, real-world leaf photo to sanity-check generalization beyond the validation set.
4. **Dashboard (`dashboard/`)** — A Streamlit app that loads the trained models and runs live inference on any uploaded leaf image.

## 🧠 Models Compared

All four models use the same recipe: **frozen ImageNet-pretrained convolutional base + custom dense classification head**, trained with `Adam` (`lr=1e-4`) and `categorical_crossentropy`.

| Model | Input Size | Head Architecture |
|---|---|---|
| ResNet50 | 256×256 | GAP → Dense(512, relu) → Dense(39, softmax) |
| VGG16 | 224×224 | Flatten → Dense(256, relu) → Dropout(0.5) → Dense(39, softmax) |
| MobileNetV2 | 224×224 | GAP → Dense(256, relu) → Dropout(0.5) → Dense(39, softmax) |
| EfficientNetB0 | 224×224 | GAP → Dense(256, relu) → Dropout(0.5) → Dense(39, softmax) |

## 📈 Results

Results below are reconstructed directly from the executed cell outputs in `notebooks/01_kaggle_training.ipynb`.

| Model | Val. Accuracy | Val. Loss | Model Size | Verdict |
|---|---|---|---|---|
| ResNet50 | ~73.4% | 0.91 | Large | Baseline; underfit relative to others |
| **VGG16** | **~96.6%** | — | Large (~500MB+) | Highest accuracy, but heaviest model |
| **MobileNetV2** | **~97.0%** | 0.11 | **~12.5MB** | Best accuracy-to-size tradeoff — **recommended for deployment** |
| EfficientNetB0 | ~9.9% | 3.37 | Medium | Did not converge in this run — see note below |

> ⚠️ **Note on EfficientNetB0:** this checkpoint did not converge (near-random accuracy, predictions collapsed to one dominant class). A likely cause is the `ImageDataGenerator(rescale=1./255)` pipeline double-normalizing input that `EfficientNetB0` already rescales internally — worth revisiting with `tf.keras.applications.efficientnet.preprocess_input` instead. **If your final thesis report has different/corrected EfficientNet numbers, replace this row with the authoritative ones.**

📌 **Bottom line:** VGG16 edges out on raw validation accuracy, but **MobileNetV2 is the practical winner** — comparable accuracy at a fraction of the size, making it the model used in the dashboard by default.

*(Optional: add your confusion matrices / accuracy-vs-epoch plots as PNGs to `results/` and embed them here, e.g. `![Confusion Matrix](results/vgg16_confusion_matrix.png)`)*

## 📁 Repository Structure

```
plant-disease-classification/
├── README.md                          # You are here
├── LICENSE
├── requirements.txt                    # Full project dependencies (training + dashboard)
├── .gitignore
│
├── notebooks/
│   ├── 01_kaggle_training.ipynb        # All 4 models trained on Kaggle GPU
│   ├── 02_eda_and_model_comparison.ipynb  # Dataset EDA + model evaluation (Colab)
│   └── 03_real_image_inference_test.ipynb # Real-world image sanity test (Colab)
│
├── dashboard/                          # Deployable inference demo
│   ├── app.py                          # Streamlit multi-model dashboard
│   ├── class_names.py                  # 39-class label mapping
│   ├── requirements.txt                # Minimal deps to run just the dashboard
│   └── README.md                       # Dashboard-specific instructions
│
├── models/                             # Trained .keras weights (NOT committed to git — see below)
│   └── README.md                       # Where to download / how to regenerate weights
│
├── results/                            # Plots, confusion matrices, sample predictions
│   └── README.md
│
└── docs/
    └── thesis_summary.md               # Condensed write-up for portfolio/recruiter reading
```

## ⚙️ Setup & Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/plant-disease-classification.git
cd plant-disease-classification

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

## 📓 Running the Notebooks

- **`01_kaggle_training.ipynb`** — Upload to [Kaggle](https://www.kaggle.com/code), attach the [PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease), enable GPU (Settings → Accelerator → GPU), and run all cells. Models save to `/kaggle/working/`.
- **`02_eda_and_model_comparison.ipynb`** / **`03_real_image_inference_test.ipynb`** — Open in [Google Colab](https://colab.research.google.com/), mount Google Drive, and point `data_dir` / model paths to wherever you stored the dataset and `.keras` checkpoints.

## 🖥️ Running the Dashboard

```bash
cd dashboard
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints (typically `http://localhost:8501`), upload a leaf image, and compare predictions across all available models. See [`dashboard/README.md`](dashboard/README.md) for details on configuring model paths.

## 💾 Model Weights

Trained `.keras` files (ResNet50, VGG16 especially) are **too large for a normal git push** (GitHub blocks files >100MB; recommends Git LFS above 50MB). This repo ships the **code only**. To get the dashboard fully working, choose one:

1. **Git LFS** — `git lfs track "*.keras"` then commit the models normally.
2. **Hugging Face Hub** (recommended) — upload models to a HF model repo (free), and download them in `dashboard/app.py` via `huggingface_hub.hf_hub_download()`.
3. **Google Drive link** — host the `.keras` files on Drive and add a `gdown` download step to `models/README.md`.

See [`models/README.md`](models/README.md) for exact filenames the dashboard expects.

## 🛠️ Tech Stack

`Python` · `TensorFlow / Keras` · `NumPy` · `Pandas` · `Matplotlib` / `Seaborn` · `scikit-learn` · `Streamlit` · `Kaggle Notebooks (GPU)` · `Google Colab`

## 🔭 Future Work

- Re-train EfficientNetB0 with correct `preprocess_input` and fine-tune top layers (unfreeze last N blocks) for a fairer comparison.
- Add Grad-CAM visualizations to interpret model predictions.
- Convert the recommended MobileNetV2 model to `.tflite` for genuine mobile/edge deployment.
- Deploy the Streamlit dashboard publicly (Streamlit Community Cloud / Hugging Face Spaces).

## 👤 Author & Acknowledgements

**Galib Amin**
B.Tech + M.Tech (IDP), Computer Science — JNTUH University College of Engineering, Science and Technology, Hyderabad

Thesis supervised by **Dr. Athota Kavitha**.

Dataset: [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) (Mohanty et al.)

---

⭐ If you find this project useful, consider giving it a star!
